"""IncidentRAG FastAPI Application.

Endpoints:
  POST /alerts/ingest          — ingest a raw alert, trigger pipeline
  GET  /assessments/{id}       — retrieve a completed assessment
  GET  /approvals              — list pending approval requests
  POST /approvals/{id}/approve — approve an action
  POST /approvals/{id}/reject  — reject an action
  POST /runbooks/ingest        — upload a runbook markdown file
  POST /resolved/{incident_id} — mark incident resolved, trigger feedback
  GET  /eval/run               — run RAGAS evaluation on stored cases
  GET  /health                 — health check
"""
from __future__ import annotations

import logging
import os
import secrets
import tempfile
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from incidentrag.api.github_router import router as github_router
from incidentrag.core.models import IncidentAssessment, RawAlert
from incidentrag.core.settings import settings
from incidentrag.feedback.loop import EvalSetExpander
from incidentrag.feedback.postmortem_generator import PostmortemGenerator
from incidentrag.graph.layer import GraphLayer
from incidentrag.ingestion.pipeline import IngestionPipeline
from incidentrag.pipeline import IncidentRAGPipeline, PipelineResult
from incidentrag.retrieval.bm25_index import BM25Index
from incidentrag.retrieval.dense_index import DenseIndex

logger = logging.getLogger(__name__)

app = FastAPI(
    title="IncidentRAG",
    description="Production-grade Incident Response RAG System",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.cors_allowed_origins.split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(github_router)

# ── Singletons (initialized at startup) ──────────────────────────────────────
_bm25: BM25Index | None = None
_dense: DenseIndex | None = None
_graph: GraphLayer | None = None
_pipeline: IncidentRAGPipeline | None = None
_approval_gate: Any | None = None

# In-memory store for completed assessments
_assessments: dict[str, IncidentAssessment] = {}
_alerts: dict[str, RawAlert] = {}
app.state.assessment_store = _assessments
app.state.alert_store = _alerts
app.state.pipeline = None


@app.middleware("http")
async def require_api_key(request: Request, call_next: Any) -> Any:
    """Require X-API-Key when INCIDENTRAG_API_KEY is configured."""
    public_paths = {"/health", "/ready", "/docs", "/openapi.json", "/redoc"}
    configured = (
        settings.incidentrag_api_key.get_secret_value()
        if settings.incidentrag_api_key
        else ""
    )
    if configured and request.url.path not in public_paths:
        supplied = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(supplied, configured):
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)


@app.on_event("startup")
async def startup() -> None:
    global _bm25, _dense, _graph, _pipeline, _approval_gate
    logger.info("Initializing IncidentRAG services...")

    _bm25 = BM25Index()
    _dense = DenseIndex()
    _graph = GraphLayer()

    try:
        await _graph.setup()
        await _dense.ensure_collection()
        stored = await _dense.list_chunks()
        if stored:
            _bm25.build(stored)
    except Exception as exc:
        logger.warning("Startup: infrastructure not fully ready: %s", exc)

    _pipeline = IncidentRAGPipeline(
        bm25=_bm25,
        dense=_dense,
        graph=_graph,
        dry_run=not settings.execution_enabled,
    )
    _approval_gate = _pipeline.approval
    app.state.pipeline = _pipeline
    logger.info("IncidentRAG ready")


@app.on_event("shutdown")
async def shutdown() -> None:
    if _graph:
        await _graph.close()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "incidentrag"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    checks = {"pipeline": _pipeline is not None}
    if _dense is not None:
        try:
            await _dense.ensure_collection()
            checks["qdrant"] = True
        except Exception:
            checks["qdrant"] = False
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.post("/alerts/ingest", response_model=dict)
async def ingest_alert(alert: RawAlert) -> dict[str, Any]:
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result: PipelineResult = await _pipeline.process(alert)
    _assessments[result.assessment.incident_id] = result.assessment
    _alerts[result.assessment.incident_id] = alert

    all_claims = [result.assessment.root_cause] + result.assessment.contributing_factors
    grounding_passed = any(c.verified for c in all_claims)

    return {
        "incident_id": result.assessment.incident_id,
        "confidence": result.assessment.overall_confidence,
        "grounding_passed": grounding_passed,
        "claims_count": len(all_claims),
        "actions_count": len(result.assessment.proposed_actions),
        "approval_requests": [
            {"id": r.request_id, "status": r.status, "action": r.action.description}
            for r in result.approval_requests
        ],
        "cost": result.cost_summary,
    }


@app.get("/assessments/{incident_id}")
async def get_assessment(incident_id: str) -> IncidentAssessment:
    assessment = _assessments.get(incident_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return assessment


@app.get("/assessments/{incident_id}/postmortem", response_class=PlainTextResponse)
async def get_postmortem(incident_id: str) -> str:
    assessment = _assessments.get(incident_id)
    alert = _alerts.get(incident_id)
    if not assessment or not alert:
        raise HTTPException(status_code=404, detail="Incident context not found")
    return PostmortemGenerator().generate(assessment, alert)


@app.get("/approvals")
async def list_approvals() -> list[dict]:
    if not _approval_gate:
        return []
    return [
        {
            "id": r.request_id,
            "incident_id": r.incident_id,
            "description": r.action.description,
            "risk_level": r.action.risk_level.value,
            "command": r.action.command,
            "status": r.status,
            "created_at": r.created_at.isoformat(),
        }
        for r in _approval_gate.list_pending()
    ]


@app.post("/approvals/{request_id}/approve")
async def approve_action(
    request_id: str, approver: str = "api_user", notes: str = ""
) -> dict:
    if not _approval_gate:
        raise HTTPException(status_code=503, detail="Gate not initialized")
    try:
        req = _approval_gate.approve(request_id, approver, notes)
        return {"status": req.status, "request_id": req.request_id}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/approvals/{request_id}/reject")
async def reject_action(
    request_id: str, approver: str = "api_user", notes: str = ""
) -> dict:
    if not _approval_gate:
        raise HTTPException(status_code=503, detail="Gate not initialized")
    try:
        req = _approval_gate.reject(request_id, approver, notes)
        return {"status": req.status, "request_id": req.request_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/approvals/{request_id}/execute")
async def execute_action(request_id: str) -> dict[str, Any]:
    if not settings.execution_enabled:
        raise HTTPException(status_code=403, detail="Execution is disabled")
    if not _approval_gate or not _pipeline:
        raise HTTPException(status_code=503, detail="Execution services are unavailable")
    req = _approval_gate.get(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if not req.action.command:
        raise HTTPException(status_code=400, detail="Action has no executable command")
    result = await _pipeline.executor.execute(
        request_id,
        req.action.command,
        req.status.value,
    )
    return result.model_dump(mode="json")


@app.post("/runbooks/ingest")
async def ingest_runbook(file: UploadFile) -> dict:
    if _dense is None or _bm25 is None:
        raise HTTPException(status_code=503, detail="Retrieval indexes are unavailable")
    content = await file.read()
    pipeline = IngestionPipeline(indexer=_dense)
    # Write to temp file so IngestionPipeline.ingest_file() can process it
    suffix = os.path.splitext(file.filename or "runbook.md")[1] or ".md"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = await pipeline.ingest_file(tmp_path)
    finally:
        with suppress(OSError):
            os.unlink(tmp_path)
    if not result.success:
        raise HTTPException(status_code=422, detail="Runbook ingestion failed")
    for chunk in result.chunks:
        _bm25.add(chunk)
    return {"chunks_created": len(result.chunks), "source": file.filename}


@app.post("/resolved/{incident_id}")
async def mark_resolved(incident_id: str, ground_truth: str = "") -> dict:
    assessment = _assessments.get(incident_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    await _pipeline.on_resolved(
        assessment,
        alert=_alerts.get(incident_id),
        ground_truth=ground_truth or None,
    )
    return {"status": "feedback_recorded", "incident_id": incident_id}


@app.get("/eval/run")
async def run_evaluation() -> dict:
    expander = EvalSetExpander()
    cases = expander.load()
    if not cases:
        return {"message": "No evaluation cases found. Resolve incidents to build eval set."}

    return {"message": f"Found {len(cases)} eval cases. Pair with assessments to run RAGAS."}
