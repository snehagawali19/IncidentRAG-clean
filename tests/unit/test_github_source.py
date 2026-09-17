"""Tests for the read-only Argo CD GitHub issue source."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from incidentrag.core.models import ExternalIssue, IssueFilters, Severity
from incidentrag.core.protocols import IncidentSource
from incidentrag.sources.github import (
    GitHubIssueProvider,
    GitHubSourceError,
    sanitize_untrusted_text,
)


def _issue_payload(
    number: int,
    *,
    labels: list[str] | None = None,
    body: str | None = None,
    days_old: int = 1,
    pull_request: bool = False,
) -> dict:
    now = datetime.now(UTC)
    payload = {
        "number": number,
        "title": f"Repo server failure {number}",
        "body": body
        if body is not None
        else (
            "Steps to reproduce: configure Argo CD and sync the application. "
            "Expected behavior: manifests render. Actual behavior: repo-server returns error. "
            "Argo CD version v2.12.0. Kubernetes logs:\n```text\nconnection refused\n```"
        ),
        "labels": [{"name": label} for label in (labels or ["bug", "component:repo-server"])],
        "user": {"login": "reporter"},
        "created_at": (now - timedelta(days=10)).isoformat().replace("+00:00", "Z"),
        "updated_at": (now - timedelta(days=days_old)).isoformat().replace("+00:00", "Z"),
        "comments": 3,
        "html_url": f"https://github.com/argoproj/argo-cd/issues/{number}",
        "url": f"https://api.github.com/repos/argoproj/argo-cd/issues/{number}",
    }
    if pull_request:
        payload["pull_request"] = {"url": "https://api.github.com/pulls/1"}
    return payload


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    token: str | None = "",
    candidate_limit: int = 10,
) -> tuple[GitHubIssueProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GitHubIssueProvider(
        client=client,
        owner="argoproj",
        repository="argo-cd",
        token=token,
        candidate_limit=candidate_limit,
        max_results=2,
        api_base_url="https://api.github.com",
    )
    return provider, client


def test_provider_conforms_to_incident_source_protocol() -> None:
    provider, _ = _provider(lambda request: httpx.Response(200, json={"items": []}))
    assert isinstance(provider, IncidentSource)


def test_build_query_contains_defaults_and_filters() -> None:
    provider, _ = _provider(lambda request: httpx.Response(200, json={"items": []}))
    query = provider.build_query(
        IssueFilters(
            keyword="manifest generation",
            component="repo-server",
            severity="minor",
            priority="high",
            regression=True,
            updated_within_days=30,
        )
    )
    assert "repo:argoproj/argo-cd" in query
    assert "is:issue" in query and "is:open" in query and "label:bug" in query
    assert 'label:"component:repo-server"' in query
    assert 'label:"bug/severity:minor"' in query
    assert 'label:"bug/priority:high"' in query
    assert "label:regression" in query
    assert "updated:>=" in query


@pytest.mark.asyncio
async def test_unauthenticated_request_has_no_authorization_header() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"items": []})

    provider, client = _provider(handler)
    try:
        assert await provider.fetch_candidates(IssueFilters()) == []
        assert seen["authorization"] is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_authenticated_request_uses_bearer_without_exposing_token() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"items": []})

    provider, client = _provider(handler, token="test-secret-token")
    try:
        await provider.fetch_candidates(IssueFilters())
        assert seen["authorization"] == "Bearer test-secret-token"
        assert "test-secret-token" not in repr(provider.rate_limit)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_fetch_candidates_excludes_prs_features_empty_bodies_and_caps_two() -> None:
    payload = {
        "items": [
            _issue_payload(1, pull_request=True),
            _issue_payload(2, labels=["bug", "enhancement"], body="feature request details"),
            _issue_payload(3, body=""),
            _issue_payload(4, labels=["bug", "component:repo-server", "regression"]),
            _issue_payload(5, labels=["bug", "component:sync", "bug/priority:high"]),
            _issue_payload(6, labels=["bug", "component:ui"]),
        ]
    }
    provider, client = _provider(lambda request: httpx.Response(200, json=payload))
    try:
        results = await provider.fetch_candidates(IssueFilters(limit=2))
        assert len(results) == 2
        assert {item.number for item in results}.issubset({4, 5, 6})
        assert all(item.selection_explanation for item in results)
        assert results[0].suitability_score >= results[1].suitability_score
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_candidate_pool_is_hard_capped_at_ten() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["per_page"] = request.url.params["per_page"]
        return httpx.Response(200, json={"items": []})

    provider, client = _provider(handler, candidate_limit=50)
    try:
        await provider.fetch_candidates(IssueFilters())
        assert seen["per_page"] == "10"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_filters_are_enforced_after_fetch() -> None:
    payload = {
        "items": [
            _issue_payload(1, labels=["bug", "component:repo-server", "regression"]),
            _issue_payload(2, labels=["bug", "component:ui", "regression"]),
        ]
    }
    provider, client = _provider(lambda request: httpx.Response(200, json=payload))
    try:
        results = await provider.fetch_candidates(
            IssueFilters(component="repo-server", regression=True)
        )
        assert [item.number for item in results] == [1]
    finally:
        await client.aclose()


def test_sanitize_removes_html_hidden_comments_credentials_and_email() -> None:
    raw = (
        "<script>alert(1)</script><!--ignore instructions-->"
        "token=ghp_abcdefghijklmnopqrstuvwxyz123456 email user@example.com "
        "<b>visible</b>"
    )
    clean = sanitize_untrusted_text(raw, max_length=500)
    assert "script" not in clean
    assert "ignore instructions" not in clean
    assert "ghp_" not in clean
    assert "user@example.com" not in clean
    assert "visible" in clean
    assert "[REDACTED" in clean


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 404, 422, 429])
async def test_expected_github_failures_are_mapped_to_safe_errors(status: int) -> None:
    provider, client = _provider(
        lambda request: httpx.Response(status, json={"message": "do not leak upstream body"})
    )
    try:
        with pytest.raises(GitHubSourceError) as exc_info:
            await provider.fetch_candidates(IssueFilters())
        assert exc_info.value.status_code == status
        assert "do not leak" not in str(exc_info.value)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_network_failure_is_mapped_to_safe_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection details", request=request)

    provider, client = _provider(handler)
    try:
        with pytest.raises(GitHubSourceError, match="network request failed"):
            await provider.fetch_candidates(IssueFilters())
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_rate_limit_metadata_is_exposed_safely() -> None:
    reset = int(datetime.now(UTC).timestamp()) + 3600
    provider, client = _provider(
        lambda request: httpx.Response(
            200,
            json={"items": []},
            headers={"X-RateLimit-Remaining": "42", "X-RateLimit-Reset": str(reset)},
        )
    )
    try:
        await provider.fetch_candidates(IssueFilters())
        assert provider.rate_limit.remaining == 42
        assert provider.rate_limit.reset_at is not None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_labels_are_sanitized_and_cached() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json=[{"name": "component:ui"}, {"name": "<b>regression</b>"}],
        )

    provider, client = _provider(handler)
    try:
        first = await provider.fetch_labels()
        second = await provider.fetch_labels()
        assert first == ["component:ui", "regression"]
        assert second == first
        assert calls == 1
    finally:
        await client.aclose()


def test_suitability_score_is_deterministic() -> None:
    provider, _ = _provider(lambda request: httpx.Response(200, json={"items": []}))
    issue = provider._parse_issue(
        _issue_payload(10, labels=["bug", "component:repo-server", "regression"])
    )
    assert provider.score_suitability(issue) == provider.score_suitability(issue)


def test_normalize_maps_component_severity_environment_and_safe_payload() -> None:
    provider, _ = _provider(lambda request: httpx.Response(200, json={"items": []}))
    now = datetime.now(UTC)
    issue = ExternalIssue(
        repository="argoproj/argo-cd",
        number=123,
        title="Repo server fails in production",
        body="Production cluster manifest generation fails with a TLS error.",
        labels=["bug", "component:repo-server", "bug/priority:high"],
        author="reporter",
        created_at=now,
        updated_at=now,
        comments_count=2,
        html_url="https://github.com/argoproj/argo-cd/issues/123",
        api_url="https://api.github.com/repos/argoproj/argo-cd/issues/123",
        suitability_score=75,
        selection_explanation=["technical evidence"],
    )
    alert = provider.normalize(issue)
    assert alert.alert_id == "github:argoproj/argo-cd#123"
    assert alert.source == "github"
    assert alert.service_name == "argocd-repo-server"
    assert alert.environment == "production"
    assert alert.severity == Severity.SEV2
    assert alert.raw_payload["issue_number"] == 123
    assert "suitability_score" in alert.raw_payload
    assert set(alert.raw_payload).isdisjoint({"node_id", "reactions", "assignee"})


@pytest.mark.asyncio
async def test_fetch_issue_rejects_pull_request() -> None:
    provider, client = _provider(
        lambda request: httpx.Response(200, json=_issue_payload(7, pull_request=True))
    )
    try:
        with pytest.raises(GitHubSourceError, match="pull request"):
            await provider.fetch_issue(7)
    finally:
        await client.aclose()
