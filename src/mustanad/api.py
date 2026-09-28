"""FastAPI application.

Routes
------
``GET  /healthz``            liveness, unauthenticated
``GET  /readyz``             readiness plus corpus size, unauthenticated
``POST /v1/ask``             ask a question, get a cited answer
``POST /v1/ingest/text``     index a raw string
``POST /v1/ingest/upload``   index an uploaded file
``GET  /v1/documents``       list the corpus
``DELETE /v1/documents/{id}``remove a document and its index entries
``GET  /``                   a minimal demo page

Everything under ``/v1`` requires the ``X-API-Key`` header unless
``MUSTANAD_AUTH_REQUIRED=false``.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .index.store import DocumentStore
from .logging_setup import configure_logging
from .providers import BudgetExceededError, ProviderError
from .security import SlidingWindowRateLimiter, key_fingerprint, verify_api_key
from .service import AskResult, IngestionError, MustanadService

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


# --------------------------------------------------------------------------- schemas


class AskRequestModel(BaseModel):
    """A question to answer from the indexed corpus."""

    question: str = Field(min_length=2, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationModel(BaseModel):
    index: int
    document_id: int
    document_title: str
    page: int | None
    label: str
    excerpt: str
    score: float


class AskResponseModel(BaseModel):
    question: str
    answer: str
    grounded: bool = Field(
        description="False when no passage supported a confident answer. When false, `answer` "
        "is an explicit 'not found' message, not a guess."
    )
    confidence: float = Field(description="Rerank score of the best passage, in [0, 1].")
    provider: str
    model: str | None
    warnings: list[str]
    citations: list[CitationModel]


class IngestTextRequestModel(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    source: str = Field(default="api:text", max_length=500)


class IngestResponseModel(BaseModel):
    document_id: int
    title: str
    language: str
    chunk_count: int
    already_indexed: bool


class DocumentModel(BaseModel):
    id: int
    title: str
    source: str
    language: str
    chunk_count: int
    token_count: int
    ingested_at: str


class HealthModel(BaseModel):
    status: str
    provider: str
    documents: int
    chunks: int


# ----------------------------------------------------------------------- dependencies
#
# These live at module scope rather than inside ``create_app`` on purpose. With
# ``from __future__ import annotations`` every annotation is a string, and FastAPI resolves
# route annotations against the *module* namespace -- so an ``Annotated`` alias defined inside
# the factory is unresolvable and every route silently degrades to treating its dependency as
# a request body field. Shared state therefore travels on ``app.state`` instead of a closure.


def get_service(request: Request) -> MustanadService:
    """The application service, created once per process in the lifespan handler."""
    service: MustanadService = request.app.state.service
    return service


def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
) -> str:
    """Authenticate the caller and apply the per-identity rate limit.

    Returns:
        The caller's identity label -- an API-key fingerprint, or ``"anonymous"`` when auth is
        disabled -- which is safe to log because it is not reversible to the key.

    Raises:
        HTTPException: 401 when the key is missing or unknown, 429 when the caller is over
            its rate limit.
    """
    settings: Settings = request.app.state.settings
    limiter: SlidingWindowRateLimiter = request.app.state.limiter

    if not settings.auth_required:
        identity = "anonymous"
    else:
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail=f"missing {API_KEY_HEADER} header",
                headers={"WWW-Authenticate": API_KEY_HEADER},
            )
        if not verify_api_key(x_api_key, settings.api_key_hashes):
            logger.warning(
                "rejected api key",
                extra={"fingerprint": key_fingerprint(x_api_key), "path": request.url.path},
            )
            raise HTTPException(status_code=401, detail="invalid API key")
        identity = key_fingerprint(x_api_key)

    allowed, retry_after = limiter.check(identity)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(int(retry_after), 1))},
        )
    return identity


Service = Annotated[MustanadService, Depends(get_service)]
Authenticated = Annotated[str, Depends(require_api_key)]


# --------------------------------------------------------------------------- wiring


def _to_response(result: AskResult) -> AskResponseModel:
    return AskResponseModel(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        confidence=round(result.confidence, 4),
        provider=result.provider,
        model=result.model,
        warnings=list(result.warnings),
        citations=[
            CitationModel(
                index=citation.index,
                document_id=citation.document_id,
                document_title=citation.document_title,
                page=citation.page,
                label=citation.label,
                excerpt=citation.excerpt,
                score=citation.score,
            )
            for citation in result.citations
        ],
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    Accepting ``settings`` makes the app testable without touching the process environment.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    limiter = SlidingWindowRateLimiter(limit=resolved.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = DocumentStore(resolved.db_path)
        service = MustanadService(resolved, store)
        app.state.service = service
        logger.info(
            "mustanad ready",
            extra={"provider": resolved.provider, "db": str(resolved.db_path)},
        )
        try:
            yield
        finally:
            service.close()

    app = FastAPI(
        title="mustanad",
        version="0.1.0",
        summary="Grounded question answering over your own documents, in Arabic and English.",
        description=(
            "Every answer is accompanied by the passages it came from. When no passage "
            "supports an answer, the API says so instead of guessing."
        ),
        lifespan=lifespan,
    )
    # Set before any request can arrive, so the module-level dependencies above can rely on it.
    app.state.settings = resolved
    app.state.limiter = limiter

    if resolved.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=[API_KEY_HEADER, "Content-Type"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable) -> Response:
        """Attach a request id and log one structured line per request."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response

    # ----------------------------------------------------------------- health

    @app.get("/healthz", response_model=HealthModel, tags=["health"])
    def healthz(service: Service) -> HealthModel:
        stats = service.store.collection_stats()
        return HealthModel(
            status="ok",
            provider=service.provider.name,
            documents=len(service.list_documents()),
            chunks=stats.chunk_count,
        )

    @app.get("/readyz", response_model=HealthModel, tags=["health"])
    def readyz(service: Service, response: Response) -> HealthModel:
        stats = service.store.collection_stats()
        # An empty corpus is a live service that cannot yet do its job: report it as
        # not-ready so an orchestrator does not send traffic to it.
        if stats.chunk_count == 0:
            response.status_code = 503
        return HealthModel(
            status="ok" if stats.chunk_count else "empty-corpus",
            provider=service.provider.name,
            documents=len(service.list_documents()),
            chunks=stats.chunk_count,
        )

    # ----------------------------------------------------------------- ask

    @app.post("/v1/ask", response_model=AskResponseModel, tags=["ask"])
    def ask(body: AskRequestModel, service: Service, identity: Authenticated) -> AskResponseModel:
        try:
            result = service.ask(body.question, top_k=body.top_k)
        except BudgetExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info(
            "ask",
            extra={
                "identity": identity,
                "grounded": result.grounded,
                "confidence": round(result.confidence, 3),
                "citations": len(result.citations),
            },
        )
        return _to_response(result)

    # ----------------------------------------------------------------- ingest

    @app.post(
        "/v1/ingest/text",
        response_model=IngestResponseModel,
        status_code=201,
        tags=["ingest"],
    )
    def ingest_text(
        body: IngestTextRequestModel, service: Service, identity: Authenticated
    ) -> IngestResponseModel:
        if len(body.text.encode("utf-8")) > resolved.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"text exceeds the {resolved.max_upload_bytes} byte limit",
            )
        try:
            result = service.ingest_text(body.text, source=body.source, title=body.title)
        except IngestionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponseModel(
            document_id=result.document.id,
            title=result.document.title,
            language=result.document.language,
            chunk_count=result.chunk_count,
            already_indexed=result.already_indexed,
        )

    @app.post(
        "/v1/ingest/upload",
        response_model=IngestResponseModel,
        status_code=201,
        tags=["ingest"],
    )
    async def ingest_upload(
        service: Service,
        identity: Authenticated,
        file: Annotated[UploadFile, File()],
    ) -> IngestResponseModel:
        filename = Path(file.filename or "upload")
        suffix = filename.suffix.lower()
        if suffix not in resolved.allowed_suffixes:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported file type {suffix!r}; allowed: "
                f"{', '.join(resolved.allowed_suffixes)}",
            )

        payload = await file.read(resolved.max_upload_bytes + 1)
        if len(payload) > resolved.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds the {resolved.max_upload_bytes} byte limit",
            )
        if not payload:
            raise HTTPException(status_code=422, detail="uploaded file is empty")

        # The loaders work on paths (pypdf and python-docx both want a file), so the upload is
        # written to a private temporary file. Only the *original name* is used for the title;
        # the path itself is generated here, so a crafted filename cannot traverse anywhere.
        with tempfile.TemporaryDirectory(prefix="mustanad-upload-") as directory:
            temporary = Path(directory) / f"upload{suffix}"
            temporary.write_bytes(payload)
            try:
                result = service.ingest_path(temporary, title=filename.stem or filename.name)
            except IngestionError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        return IngestResponseModel(
            document_id=result.document.id,
            title=result.document.title,
            language=result.document.language,
            chunk_count=result.chunk_count,
            already_indexed=result.already_indexed,
        )

    # ----------------------------------------------------------------- corpus

    @app.get("/v1/documents", response_model=list[DocumentModel], tags=["documents"])
    def list_documents(service: Service, identity: Authenticated) -> list[DocumentModel]:
        return [
            DocumentModel(
                id=document.id,
                title=document.title,
                source=document.source,
                language=document.language,
                chunk_count=document.n_chunks,
                token_count=document.n_tokens,
                ingested_at=document.ingested_at,
            )
            for document in service.list_documents()
        ]

    @app.delete("/v1/documents/{document_id}", status_code=204, tags=["documents"])
    def delete_document(document_id: int, service: Service, identity: Authenticated) -> Response:
        if not service.delete_document(document_id):
            raise HTTPException(status_code=404, detail=f"no document with id {document_id}")
        return Response(status_code=204)

    # ----------------------------------------------------------------- demo page

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo_page() -> HTMLResponse:
        return HTMLResponse(DEMO_HTML)

    @app.exception_handler(500)
    async def internal_error(request: Request, exc: Exception) -> JSONResponse:
        # Never leak a traceback or a document excerpt to a caller.
        logger.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    return app


DEMO_HTML = """<!doctype html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mustanad — ask your documents</title>
<style>
  :root { color-scheme: light dark; --fg:#14181d; --bg:#fbfbfd; --muted:#5b6673;
          --line:#dfe3e8; --accent:#1f6feb; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8ecf1; --bg:#14181d; --muted:#98a2ae; --line:#2b323b; --accent:#589bff; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:2rem 1rem; background:var(--bg); color:var(--fg);
         font:16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 52rem; margin:0 auto; }
  h1 { font-size:1.5rem; margin:0 0 .25rem; }
  p.lede { color:var(--muted); margin:0 0 1.5rem; }
  form { display:flex; gap:.5rem; flex-wrap:wrap; }
  input, button { font:inherit; padding:.6rem .8rem; border-radius:8px;
                  border:1px solid var(--line); background:var(--bg); color:var(--fg); }
  input[type=text] { flex:1 1 20rem; }
  button { background:var(--accent); color:#fff; border-color:transparent; cursor:pointer; }
  button:disabled { opacity:.6; cursor:progress; }
  #answer { margin-top:1.5rem; padding:1rem; border:1px solid var(--line); border-radius:10px;
            white-space:pre-wrap; }
  #answer:empty { display:none; }
  .tag { display:inline-block; font-size:.75rem; padding:.1rem .5rem; border-radius:999px;
         border:1px solid var(--line); color:var(--muted); margin-inline-end:.4rem; }
  ol { padding-inline-start:1.4rem; }
  li { margin-bottom:.75rem; }
  li .src { color:var(--muted); font-size:.85rem; }
  [dir=rtl] { text-align:right; }
</style>
</head>
<body>
<main>
  <h1>mustanad</h1>
  <p class="lede">Ask a question about the indexed documents. Every answer shows the
     passages it came from. Arabic and English both work.</p>
  <form id="f">
    <input type="text" id="q" placeholder="e.g. How many days of annual leave?" required>
    <input type="text" id="k" placeholder="API key" size="14">
    <button id="go" type="submit">Ask</button>
  </form>
  <div id="answer"></div>
  <ol id="cites"></ol>
</main>
<script>
const form = document.getElementById('f');
const answerBox = document.getElementById('answer');
const citeList = document.getElementById('cites');
const button = document.getElementById('go');

const rtl = (s) => /[\\u0600-\\u06FF]/.test(s);

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = document.getElementById('q').value.trim();
  if (!question) return;
  button.disabled = true;
  answerBox.textContent = 'Searching…';
  citeList.innerHTML = '';
  try {
    const headers = { 'Content-Type': 'application/json' };
    const key = document.getElementById('k').value.trim();
    if (key) headers['X-API-Key'] = key;
    const response = await fetch('/v1/ask', {
      method: 'POST', headers, body: JSON.stringify({ question })
    });
    const data = await response.json();
    if (!response.ok) {
      answerBox.textContent = 'Error: ' + (data.detail || response.status);
      return;
    }
    answerBox.dir = rtl(data.answer) ? 'rtl' : 'ltr';
    const tags = [
      data.grounded ? 'grounded' : 'no supporting passage',
      'provider: ' + data.provider,
      'confidence: ' + data.confidence
    ].map(t => '<span class="tag">' + t + '</span>').join('');
    answerBox.innerHTML = tags + '<div style="margin-top:.75rem"></div>';
    answerBox.lastChild.textContent = data.answer;
    for (const c of data.citations) {
      const li = document.createElement('li');
      li.dir = rtl(c.excerpt) ? 'rtl' : 'ltr';
      const src = document.createElement('div');
      src.className = 'src';
      src.textContent = '[' + c.index + '] ' + c.label + ' — score ' + c.score;
      const body = document.createElement('div');
      body.textContent = c.excerpt;
      li.append(src, body);
      citeList.appendChild(li);
    }
  } catch (error) {
    answerBox.textContent = 'Request failed: ' + error;
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""
