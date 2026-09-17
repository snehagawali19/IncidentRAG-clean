"""FastAPI routes for the manual, read-only Argo CD GitHub source."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from incidentrag.core.exceptions import GenerationError, RetrievalError
from incidentrag.core.models import ExternalIssue, IssueFilters, RawAlert
from incidentrag.sources.github import GitHubIssueProvider, GitHubSourceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources/github", tags=["GitHub source"])
_provider: GitHubIssueProvider | None = None

DISCLAIMER = (
    "This is an AI-assisted technical triage based on public issue content and "
    "available runbook evidence. It is not an official Argo CD maintainer diagnosis."
)


def get_github_provider() -> GitHubIssueProvider:
    global _provider
    if _provider is None:
        _provider = GitHubIssueProvider()
    return _provider


def get_analysis_pipeline(request: Request) -> Any:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Assessment pipeline is unavailable")
    return pipeline


def get_assessment_store(request: Request) -> dict[str, Any]:
    return cast(dict[str, Any], request.app.state.assessment_store)


def _rate_limit(provider: GitHubIssueProvider) -> dict[str, Any]:
    return provider.rate_limit.model_dump(mode="json")


def _provider_http_error(exc: GitHubSourceError) -> HTTPException:
    if exc.kind == "timeout":
        return HTTPException(status_code=504, detail="GitHub request timed out")
    if exc.kind == "network":
        return HTTPException(status_code=503, detail="GitHub is currently unreachable")
    if exc.kind == "rate_limit":
        return HTTPException(status_code=429, detail="GitHub rate limit exceeded")
    if exc.status_code == 404:
        return HTTPException(status_code=404, detail="GitHub issue not found")
    if exc.status_code in {401, 403}:
        return HTTPException(status_code=exc.status_code, detail="GitHub access denied")
    return HTTPException(status_code=502, detail="GitHub returned an invalid response")


def _filters(
    *,
    keyword: str | None,
    component: str | None,
    severity: str | None,
    priority: str | None,
    regression: bool,
    updated_days: int,
    limit: int,
) -> IssueFilters:
    try:
        return IssueFilters(
            keyword=keyword,
            component=component,
            severity=severity,
            priority=priority,
            regression=regression,
            updated_within_days=updated_days,
            limit=limit,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="Invalid GitHub issue filters") from exc


async def _with_comments(
    provider: GitHubIssueProvider,
    issue: ExternalIssue,
    include_comments: bool,
) -> ExternalIssue:
    if not include_comments or issue.comments_count == 0:
        return issue
    comments = await provider.fetch_recent_comments(issue.number, limit=3)
    return issue.model_copy(update={"recent_comments": comments})


@router.get("/status")
async def github_status(
    provider: Annotated[GitHubIssueProvider, Depends(get_github_provider)],
) -> dict[str, Any]:
    return {
        "source": "github",
        "configured": bool(provider.owner and provider.repository),
        "repository": f"{provider.owner}/{provider.repository}",
        "authenticated": provider.authenticated,
        "manual_fetch_only": True,
        "candidate_limit": min(provider.candidate_limit, provider.MAX_CANDIDATES),
        "result_limit": min(provider.max_results, provider.MAX_RESULTS),
    }


@router.get("/labels")
async def github_labels(
    provider: Annotated[GitHubIssueProvider, Depends(get_github_provider)],
) -> dict[str, Any]:
    try:
        labels = await provider.fetch_label_details()
    except GitHubSourceError as exc:
        raise _provider_http_error(exc) from exc
    return {
        "repository": f"{provider.owner}/{provider.repository}",
        "labels": [label.model_dump(mode="json") for label in labels],
        "rate_limit": _rate_limit(provider),
    }


@router.get("/issues")
async def github_issues(
    provider: Annotated[GitHubIssueProvider, Depends(get_github_provider)],
    keyword: str | None = None,
    component: str | None = None,
    severity: str | None = None,
    priority: str | None = None,
    regression: bool = False,
    updated_days: int = 90,
    include_comments: bool = False,
    limit: int = 2,
) -> dict[str, Any]:
    filters = _filters(
        keyword=keyword,
        component=component,
        severity=severity,
        priority=priority,
        regression=regression,
        updated_days=updated_days,
        limit=limit,
    )
    try:
        issues = await provider.fetch_candidates(filters)
        issues = [
            await _with_comments(provider, issue, include_comments)
            for issue in issues[:2]
        ]
    except GitHubSourceError as exc:
        raise _provider_http_error(exc) from exc
    return {
        "repository": f"{provider.owner}/{provider.repository}",
        "manual_fetch": True,
        "count": len(issues),
        "issues": [issue.model_dump(mode="json") for issue in issues],
        "rate_limit": _rate_limit(provider),
    }


@router.get("/issues/{issue_number}")
async def github_issue(
    issue_number: int,
    provider: Annotated[GitHubIssueProvider, Depends(get_github_provider)],
    include_comments: Annotated[bool, Query()] = False,
) -> dict[str, Any]:
    if issue_number <= 0:
        raise HTTPException(status_code=400, detail="Issue number must be positive")
    try:
        issue = await provider.fetch_issue(issue_number)
        if not provider.is_suitable(issue):
            raise HTTPException(status_code=400, detail="Issue is not suitable for analysis")
        issue = await _with_comments(provider, issue, include_comments)
    except GitHubSourceError as exc:
        raise _provider_http_error(exc) from exc
    return {
        "issue": issue.model_dump(mode="json"),
        "rate_limit": _rate_limit(provider),
        "untrusted_external_content": True,
    }


@router.post("/issues/{issue_number}/analyze")
async def analyze_github_issue(
    issue_number: int,
    request: Request,
    provider: Annotated[GitHubIssueProvider, Depends(get_github_provider)],
    pipeline: Annotated[Any, Depends(get_analysis_pipeline)],
    assessments: Annotated[dict[str, Any], Depends(get_assessment_store)],
) -> dict[str, Any]:
    if issue_number <= 0:
        raise HTTPException(status_code=400, detail="Issue number must be positive")
    started = time.perf_counter()
    stage_history: list[dict[str, Any]] = []

    def stage(name: str, progress: int, message: str) -> None:
        stage_history.append(
            {
                "stage": name,
                "progress": progress,
                "message": message,
                "reached_at": datetime.now(UTC).isoformat(),
            }
        )

    try:
        stage("fetching_issue", 10, "Fetching selected public GitHub issue")
        issue = await provider.fetch_issue(issue_number)
        if not provider.is_suitable(issue):
            raise HTTPException(status_code=400, detail="Issue is not suitable for analysis")
        stage("normalizing", 20, "Sanitizing and normalizing untrusted issue content")
        alert: RawAlert = provider.normalize(issue)
        stage("running_incidentrag_pipeline", 30, "Running the existing IncidentRAG pipeline")
        result = await pipeline.process(alert)
        assessment = result.assessment
        assessments[assessment.incident_id] = assessment
        alert_store = getattr(request.app.state, "alert_store", None)
        if isinstance(alert_store, dict):
            alert_store[assessment.incident_id] = alert
        stage("completed", 100, "Evidence-grounded assessment completed")
    except GitHubSourceError as exc:
        raise _provider_http_error(exc) from exc
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="AI provider request timed out") from exc
    except APIConnectionError as exc:
        raise HTTPException(status_code=503, detail="AI provider is currently unreachable") from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail="AI provider rate limit exceeded") from exc
    except APIStatusError as exc:
        logger.warning(
            "AI provider rejected analysis for issue_number=%s status=%s",
            issue_number,
            exc.status_code,
        )
        if exc.status_code == 402:
            raise HTTPException(
                status_code=402,
                detail=(
                    "The AI provider requires payment or available credits. "
                    "Add credits or configure another provider, then retry."
                ),
            ) from exc
        raise HTTPException(status_code=502, detail="AI provider rejected the analysis") from exc
    except RetrievalError as exc:
        logger.info("No relevant evidence for issue_number=%s: %s", issue_number, exc)
        raise HTTPException(
            status_code=422,
            detail=(
                "No relevant runbook evidence was found for this incident. "
                "Ingest a matching runbook and retry."
            ),
        ) from exc
    except GenerationError as exc:
        logger.warning("Invalid reasoning output for issue_number=%s: %s", issue_number, exc)
        raise HTTPException(
            status_code=502,
            detail="The AI provider did not return a valid evidence-grounded assessment.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("GitHub issue analysis failed for issue_number=%s", issue_number)
        raise HTTPException(status_code=500, detail="Incident analysis failed") from exc

    claims = [assessment.root_cause, *assessment.contributing_factors]
    evidence = {
        item.chunk_id: item
        for claim in claims
        for item in claim.evidence
    }
    approval_required = any(
        request.status in {"pending", "escalated"}
        for request in result.approval_requests
    )
    return {
        "incident_id": assessment.incident_id,
        "processing": {
            "mode": "synchronous",
            "status": "completed",
            "stages": stage_history,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
        "source_issue": issue.model_dump(mode="json"),
        "facts_extracted": {
            "component": alert.service_name,
            "environment": alert.environment,
            "severity": alert.severity,
            "issue_number": issue.number,
            "repository": issue.repository,
        },
        "ai_hypotheses": {
            "likely_diagnosis": assessment.root_cause.model_dump(mode="json"),
            "possible_contributing_factors": [
                claim.model_dump(mode="json") for claim in assessment.contributing_factors
            ],
        },
        "retrieved_evidence": [item.model_dump(mode="json") for item in evidence.values()],
        "recommended_investigation": [
            action.model_dump(mode="json") for action in assessment.diagnostic_actions
        ],
        "proposed_remediation": [
            action.model_dump(mode="json") for action in assessment.proposed_actions
        ],
        "confidence": assessment.overall_confidence,
        "additional_information_required": assessment.additional_info_needed,
        "human_approval_required": approval_required,
        "assessment": assessment.model_dump(mode="json"),
        "cost": result.cost_summary,
        "disclaimer": DISCLAIMER,
    }
