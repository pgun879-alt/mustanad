"""Command-line interface.

Used for ingestion, one-off questions, and the scripted demo. It shares the service layer
with the HTTP API, so anything demonstrated here behaves identically over HTTP.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings, get_settings
from .index.store import DocumentStore
from .logging_setup import configure_logging
from .providers import ProviderError
from .service import IngestionError, MustanadService

app = typer.Typer(
    add_completion=False,
    help="Ask questions about your own documents, with citations.",
    no_args_is_help=True,
)
console = Console()


def _service(settings: Settings | None = None) -> MustanadService:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, json_output=False)
    return MustanadService(resolved, DocumentStore(resolved.db_path))


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to index.")],
    title: Annotated[str | None, typer.Option(help="Override the document title.")] = None,
) -> None:
    """Index a file, or every supported file in a directory."""
    service = _service()
    try:
        if path.is_dir():
            results = service.ingest_directory(path)
            if not results:
                console.print(f"[yellow]No supported documents found under {path}[/]")
                raise typer.Exit(code=1)
            for result in results:
                state = "already indexed" if result.already_indexed else "indexed"
                console.print(
                    f"[green]{state}[/] {result.document.title} "
                    f"(id={result.document.id}, {result.chunk_count} chunks, "
                    f"lang={result.document.language})"
                )
        else:
            result = service.ingest_path(path, title=title)
            state = "already indexed" if result.already_indexed else "indexed"
            console.print(
                f"[green]{state}[/] {result.document.title} "
                f"(id={result.document.id}, {result.chunk_count} chunks, "
                f"lang={result.document.language})"
            )
    except IngestionError as exc:
        console.print(f"[red]ingestion failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        service.close()


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, max=20)] = 5,
    show_scores: Annotated[bool, typer.Option("--scores", help="Show retrieval signals.")] = False,
) -> None:
    """Answer a question from the indexed corpus."""
    service = _service()
    try:
        result = service.ask(question, top_k=top_k)
    except ProviderError as exc:
        console.print(f"[red]provider error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    try:
        badge = "[green]grounded[/]" if result.grounded else "[yellow]no supporting passage[/]"
        console.print(
            f"\n{badge}  provider=[cyan]{result.provider}[/]  "
            f"confidence=[cyan]{result.confidence:.3f}[/]\n"
        )
        console.print(result.answer)
        if result.citations:
            console.print("\n[bold]Sources[/]")
            for citation in result.citations:
                console.print(f"  [{citation.index}] {citation.label}  (score {citation.score})")
                if show_scores:
                    console.print(f"      {citation.excerpt[:160]}")
    finally:
        service.close()


@app.command("list")
def list_documents() -> None:
    """List every indexed document."""
    service = _service()
    try:
        documents = service.list_documents()
        if not documents:
            console.print("[yellow]The corpus is empty. Run `mustanad ingest <path>` first.[/]")
            return
        table = Table(title="Indexed documents")
        for column in ("id", "title", "lang", "chunks", "tokens", "ingested"):
            table.add_column(column)
        for document in documents:
            table.add_row(
                str(document.id),
                document.title,
                document.language,
                str(document.n_chunks),
                str(document.n_tokens),
                document.ingested_at,
            )
        console.print(table)
    finally:
        service.close()


@app.command()
def delete(document_id: Annotated[int, typer.Argument(help="Document id to remove.")]) -> None:
    """Delete a document and its index entries."""
    service = _service()
    try:
        if service.delete_document(document_id):
            console.print(f"[green]deleted[/] document {document_id}")
        else:
            console.print(f"[red]no document with id {document_id}[/]")
            raise typer.Exit(code=1)
    finally:
        service.close()


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on code changes (development).")] = False,
) -> None:
    """Run the HTTP API.

    Defaults to 127.0.0.1 rather than 0.0.0.0: binding to every interface should be a
    deliberate choice, not the default a copy-pasted command gives you.
    """
    import uvicorn

    settings = get_settings()
    console.print(f"[cyan]mustanad[/] on http://{host}:{port}  (provider={settings.provider})")
    uvicorn.run("mustanad.api:create_app", host=host, port=port, reload=reload, factory=True)


@app.command()
def config() -> None:
    """Print the effective configuration, with secrets redacted."""
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001 - surface any validation error readably
        console.print(f"[red]configuration is invalid:[/] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Effective configuration")
    table.add_column("setting")
    table.add_column("value")
    redacted = {"llm_api_key", "api_keys"}
    for name, value in settings.model_dump().items():
        if name in redacted:
            shown = f"<{len(value) if isinstance(value, tuple) else 1} value(s) set>" if value else "<unset>"
        else:
            shown = str(value)
        table.add_row(name, shown)
    console.print(table)


def main() -> None:  # pragma: no cover - console entry point
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
