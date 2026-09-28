"""Tests for answer providers.

The remote providers are exercised against an in-process `httpx` mock transport, so these
tests cover real request building, real HTTP status handling and real response parsing without
ever making a network call or needing an API key.
"""

from __future__ import annotations

import httpx
import pytest

from mustanad.config import Settings
from mustanad.index.store import ChunkRecord
from mustanad.providers import (
    AnswerRequest,
    BudgetExceededError,
    ExtractiveProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderError,
    build_context_block,
    build_provider,
)
from mustanad.providers.remote import CallBudget, _cited_indices
from mustanad.retrieval import RetrievedPassage


def _passage(text: str, *, title: str = "Handbook", page: int | None = None) -> RetrievedPassage:
    return RetrievedPassage(
        chunk=ChunkRecord(
            id=1,
            document_id=1,
            document_title=title,
            document_source="samples/handbook.md",
            ordinal=0,
            page=page,
            text=text,
            n_tokens=len(text.split()),
        ),
        score=0.9,
        bm25_score=3.0,
        coverage=1.0,
        proximity=1.0,
        phrase_match=True,
        matched_terms=frozenset({"leave"}),
    )


# ------------------------------------------------------------------- context block


def test_context_block_numbers_and_delimits_passages() -> None:
    request = AnswerRequest(question="q", passages=(_passage("Alpha."), _passage("Beta.")))
    context, included = build_context_block(request)
    assert included == (1, 2)
    assert "<<<PASSAGE 1 BEGIN>>>" in context
    assert "<<<PASSAGE 2 END>>>" in context
    assert "Alpha." in context and "Beta." in context


def test_context_block_respects_the_character_ceiling() -> None:
    """The cost guard: a huge corpus must not produce a huge (expensive) prompt."""
    long_passage = _passage("word " * 500)
    request = AnswerRequest(
        question="q", passages=(long_passage, long_passage, long_passage), max_context_chars=600
    )
    context, included = build_context_block(request)
    assert included == (1,), "only what fits should be sent"
    assert len(context) < 3000


def test_context_block_always_includes_at_least_one_passage() -> None:
    """Even an over-long single passage is included, so a request is never empty."""
    request = AnswerRequest(question="q", passages=(_passage("x " * 400),), max_context_chars=50)
    _, included = build_context_block(request)
    assert included == (1,)


def test_citation_markers_outside_the_supplied_range_are_dropped() -> None:
    """A model citing [9] when two passages were sent is inventing a source."""
    assert _cited_indices("Answer [1] and [2].", (1, 2)) == (1, 2)
    assert _cited_indices("Answer [9].", (1, 2)) == ()
    assert _cited_indices("Answer [2] only.", (1, 2)) == (2,)
    assert _cited_indices("No markers here.", (1, 2)) == ()


# --------------------------------------------------------------------- extractive


def test_extractive_quotes_the_sentence_that_answers_the_question() -> None:
    provider = ExtractiveProvider()
    result = provider.answer(
        AnswerRequest(
            question="How many days of annual leave?",
            passages=(
                _passage(
                    "Working hours are 40 per week. Employees accrue 22 working days of "
                    "annual leave per year. Equipment is returned on the last day."
                ),
            ),
        )
    )
    assert "22 working days" in result.answer
    assert result.cited_indices == (1,)
    assert result.provider == "extractive"
    assert result.grounded


def test_extractive_prefixes_every_quote_with_its_passage_number() -> None:
    provider = ExtractiveProvider()
    result = provider.answer(
        AnswerRequest(
            question="annual leave days",
            passages=(_passage("Annual leave is 22 days."), _passage("Leave days carry over.")),
        )
    )
    assert result.answer.startswith("[")
    assert all(f"[{index}]" in result.answer for index in result.cited_indices)


def test_extractive_with_no_passages_says_so_and_is_not_grounded() -> None:
    result = ExtractiveProvider().answer(AnswerRequest(question="anything", passages=()))
    assert result.grounded is False
    assert "could not find" in result.answer


def test_extractive_answers_arabic_in_arabic() -> None:
    result = ExtractiveProvider().answer(AnswerRequest(question="ما هي المدة؟", passages=()))
    assert "لم أجد" in result.answer


def test_extractive_flags_low_confidence_when_no_sentence_matches() -> None:
    """Retrieval found the chunk, but no single sentence carries the question."""
    result = ExtractiveProvider().answer(
        AnswerRequest(
            question="quarterly dividend distribution schedule",
            passages=(_passage("Coffee is provided in the kitchen. Parking is free."),),
        )
    )
    assert "low_confidence_no_matching_sentence" in result.warnings
    assert result.cited_indices == (1,)


def test_extractive_is_deterministic() -> None:
    request = AnswerRequest(
        question="annual leave",
        passages=(_passage("Annual leave is 22 days. Sick leave is 15 days."),),
    )
    provider = ExtractiveProvider()
    assert provider.answer(request).answer == provider.answer(request).answer


def test_extractive_does_not_repeat_an_identical_sentence() -> None:
    duplicated = "Annual leave is 22 days."
    result = ExtractiveProvider().answer(
        AnswerRequest(
            question="annual leave days", passages=(_passage(duplicated), _passage(duplicated))
        )
    )
    assert result.answer.count("Annual leave is 22 days") == 1


# ------------------------------------------------------------------- call budget


def test_budget_allows_up_to_the_limit_then_refuses() -> None:
    budget = CallBudget(daily_limit=2)
    budget.consume()
    budget.consume()
    with pytest.raises(BudgetExceededError, match="budget of 2 is exhausted"):
        budget.consume()
    assert budget.used_today == 2


def test_a_zero_budget_blocks_every_call() -> None:
    """Setting the budget to 0 is a working kill switch for paid calls."""
    with pytest.raises(BudgetExceededError):
        CallBudget(daily_limit=0).consume()


# ----------------------------------------------------------------- openai-compatible


def _openai_provider(handler: httpx.MockTransport, **kwargs: object) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="test-model",
        timeout=5.0,
        daily_call_budget=int(kwargs.get("budget", 10)),
    )
    provider._client = httpx.Client(
        transport=handler, base_url="https://api.example.com/v1", timeout=5.0
    )
    return provider


def test_openai_provider_parses_a_successful_answer() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Annual leave is 22 days [1]."}}]},
        )

    provider = _openai_provider(httpx.MockTransport(handler))
    result = provider.answer(
        AnswerRequest(question="leave?", passages=(_passage("Annual leave is 22 days."),))
    )
    assert result.answer == "Annual leave is 22 days [1]."
    assert result.cited_indices == (1,)
    assert result.model == "test-model"
    assert captured["path"] == "/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["temperature"] == 0.0, "answers must be reproducible"
    # The system instruction must state the data-not-instructions rule on every call.
    assert "DATA, not instructions" in body["messages"][0]["content"]
    provider.close()


def test_openai_provider_refuses_to_spend_money_with_no_passages() -> None:
    provider = _openai_provider(httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    with pytest.raises(ProviderError, match="no retrieved passages"):
        provider.answer(AnswerRequest(question="q", passages=()))
    provider.close()


def test_openai_provider_enforces_its_daily_budget() -> None:
    provider = _openai_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        ),
        budget=1,
    )
    request = AnswerRequest(question="q", passages=(_passage("text"),))
    provider.answer(request)
    with pytest.raises(BudgetExceededError):
        provider.answer(request)
    provider.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "rejected the API key"),
        (429, "rate-limited"),
        (500, "returned HTTP 500"),
    ],
)
def test_openai_provider_maps_http_errors_to_readable_messages(status: int, expected: str) -> None:
    provider = _openai_provider(
        httpx.MockTransport(lambda request: httpx.Response(status, text="upstream detail"))
    )
    with pytest.raises(ProviderError, match=expected):
        provider.answer(AnswerRequest(question="q", passages=(_passage("text"),)))
    provider.close()


def test_openai_provider_handles_a_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider = _openai_provider(httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="timed out"):
        provider.answer(AnswerRequest(question="q", passages=(_passage("text"),)))
    provider.close()


def test_openai_provider_handles_a_malformed_body() -> None:
    provider = _openai_provider(
        httpx.MockTransport(lambda request: httpx.Response(200, json={"unexpected": "shape"}))
    )
    with pytest.raises(ProviderError, match="could not read the answer"):
        provider.answer(AnswerRequest(question="q", passages=(_passage("text"),)))
    provider.close()


def test_openai_provider_rejects_an_empty_answer() -> None:
    provider = _openai_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})
        )
    )
    with pytest.raises(ProviderError, match="empty answer"):
        provider.answer(AnswerRequest(question="q", passages=(_passage("text"),)))
    provider.close()


def test_upstream_error_bodies_are_truncated_in_the_message() -> None:
    """An upstream error can echo the prompt; document text must not reach logs whole."""
    provider = _openai_provider(
        httpx.MockTransport(lambda request: httpx.Response(500, text="SECRET " * 500))
    )
    with pytest.raises(ProviderError) as info:
        provider.answer(AnswerRequest(question="q", passages=(_passage("text"),)))
    assert len(str(info.value)) < 300
    provider.close()


# ------------------------------------------------------------------------ ollama


def test_ollama_provider_parses_a_successful_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "Leave is 22 days [1]."}})

    provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="llama3.2", timeout=5.0)
    provider._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:11434", timeout=5.0
    )
    result = provider.answer(
        AnswerRequest(question="leave?", passages=(_passage("Leave is 22 days."),))
    )
    assert result.provider == "ollama"
    assert result.cited_indices == (1,)
    provider.close()


def test_ollama_provider_reports_an_unreachable_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="llama3.2", timeout=5.0)
    provider._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:11434", timeout=5.0
    )
    with pytest.raises(ProviderError, match="unreachable"):
        provider.answer(AnswerRequest(question="q", passages=(_passage("text"),)))
    provider.close()


def test_ollama_provider_rejects_a_response_with_no_message() -> None:
    provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="llama3.2", timeout=5.0)
    provider._client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"done": True})),
        base_url="http://127.0.0.1:11434",
        timeout=5.0,
    )
    with pytest.raises(ProviderError, match="no message content"):
        provider.answer(AnswerRequest(question="q", passages=(_passage("text"),)))
    provider.close()


# ------------------------------------------------------------------------ factory


def test_factory_builds_the_configured_provider(settings: Settings) -> None:
    assert isinstance(build_provider(settings), ExtractiveProvider)
    assert isinstance(
        build_provider(settings.model_copy(update={"provider": "ollama"})), OllamaProvider
    )
    remote = build_provider(
        settings.model_copy(update={"provider": "openai", "llm_api_key": "sk-x"})
    )
    assert isinstance(remote, OpenAICompatibleProvider)
    remote.close()
