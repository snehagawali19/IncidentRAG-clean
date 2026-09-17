"""Typer command interface for IncidentRAG."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="incidentrag", help="Incident response RAG system", no_args_is_help=True)
console = Console()


@app.command()
def ingest(
    runbooks_dir: Annotated[
        Path, typer.Argument(help="Directory containing Markdown runbooks")
    ],
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Chunk and index runbooks in Qdrant."""
    from incidentrag.ingestion.pipeline import IngestionPipeline
    from incidentrag.retrieval.dense_index import DenseIndex

    async def _run() -> None:
        results = await IngestionPipeline(indexer=DenseIndex()).ingest_directory(
            str(runbooks_dir)
        )
        errors = [result for result in results if not result.success]
        summary = {
            "files_processed": len(results),
            "chunks_created": sum(
                len(result.chunks) for result in results if result.success
            ),
            "errors": [str(result.error) for result in errors],
        }
        if verbose:
            console.print_json(json.dumps(summary))
        else:
            console.print(
                f"[green]OK[/green] Ingested [bold]{summary['files_processed']}[/bold] "
                f"files -> [bold]{summary['chunks_created']}[/bold] chunks"
            )
        if errors:
            raise typer.Exit(code=1)

    asyncio.run(_run())


@app.command()
def analyze(
    title: str = typer.Option(..., "--title", "-t", help="Alert title"),
    body: str = typer.Option("", "--body", "-b", help="Alert body text"),
    severity: str = typer.Option("medium", "--severity", "-s"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run"),
) -> None:
    """Analyze a locally supplied incident alert end to end."""
    from incidentrag.core.models import RawAlert, Severity
    from incidentrag.graph.layer import GraphLayer
    from incidentrag.pipeline import IncidentRAGPipeline
    from incidentrag.retrieval.bm25_index import BM25Index
    from incidentrag.retrieval.dense_index import DenseIndex

    async def _run() -> None:
        try:
            selected_severity = Severity(severity.lower())
        except ValueError:
            selected_severity = Severity.MEDIUM
        alert = RawAlert(
            alert_id=f"cli-{abs(hash((title, body))) % 10_000_000}",
            source="cli",
            service_name="unknown-service",
            environment="unknown",
            severity=selected_severity,
            alert_text=f"{title}\n\n{body}".strip(),
            labels={},
            started_at=datetime.now(UTC),
        )
        graph = GraphLayer()
        pipeline = IncidentRAGPipeline(
            BM25Index(), DenseIndex(), graph, dry_run=dry_run
        )
        try:
            result = await pipeline.process(alert)
        finally:
            await graph.close()
        assessment = result.assessment
        console.print(f"[bold]Incident ID:[/bold] {assessment.incident_id}")
        console.print(
            f"[bold]Overall confidence:[/bold] {assessment.overall_confidence:.0%}"
        )
        claim = assessment.root_cause
        marker = "OK" if claim.verified else "REVIEW"
        console.print(
            f"\n[bold yellow]Root cause:[/bold yellow] [{marker}] {claim.statement}"
        )
        if assessment.proposed_actions:
            table = Table(title="Remediation actions", show_lines=True)
            table.add_column("Risk", style="bold")
            table.add_column("Description")
            table.add_column("Command")
            for action in assessment.proposed_actions:
                table.add_row(
                    action.risk_level.value,
                    action.description,
                    action.command or "",
                )
            console.print(table)

    asyncio.run(_run())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("incidentrag.api.app:app", host=host, port=port, reload=reload)


@app.command("eval-run")
def eval_run() -> None:
    """Run evaluation over stored cases."""
    from incidentrag.evaluation.runner import CIGate, RAGASRunner
    from incidentrag.feedback.loop import EvalSetExpander

    async def _run() -> None:
        cases = EvalSetExpander().load()
        if not cases:
            console.print("[yellow]No evaluation cases found.[/yellow]")
            return
        metrics = await RAGASRunner().run(cases)
        console.print_json(json.dumps(metrics.model_dump(mode="json")))
        CIGate().check(metrics)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
