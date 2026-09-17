"""Tests for the frontend's backend client boundary."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from incidentrag.approval.ui.api_client import (
    IncidentRAGAPIClient,
    IncidentRAGAPIError,
    prefer_ipv4_loopback,
)


def test_localhost_api_url_uses_ipv4_loopback() -> None:
    assert prefer_ipv4_loopback("http://localhost:8000") == "http://127.0.0.1:8000"
    client = IncidentRAGAPIClient("http://localhost:8000")
    assert client.base_url == "http://127.0.0.1:8000"


def test_find_issues_omits_empty_filters_and_sends_server_api_key() -> None:
    response = MagicMock(spec=httpx.Response)
    response.is_error = False
    response.json.return_value = {"issues": []}
    client = IncidentRAGAPIClient("http://api:8000/", "shared-secret")

    with patch(
        "incidentrag.approval.ui.api_client.httpx.request", return_value=response
    ) as request:
        client.find_issues(keyword="repo-server", component="", limit=2)

    assert request.call_args.args[:2] == ("GET", "http://api:8000/sources/github/issues")
    assert request.call_args.kwargs["params"] == {"keyword": "repo-server", "limit": 2}
    assert request.call_args.kwargs["headers"] == {"X-API-Key": "shared-secret"}


def test_backend_detail_is_shown_without_raw_http_exception() -> None:
    response = MagicMock(spec=httpx.Response)
    response.is_error = True
    response.status_code = 503
    response.json.return_value = {"detail": "GitHub is currently unreachable"}
    client = IncidentRAGAPIClient("http://api:8000")

    with (
        patch("incidentrag.approval.ui.api_client.httpx.request", return_value=response),
        pytest.raises(IncidentRAGAPIError, match="GitHub is currently unreachable") as exc,
    ):
        client.github_status()

    assert exc.value.status_code == 503


def test_analysis_uses_extended_timeout() -> None:
    response = MagicMock(spec=httpx.Response)
    response.is_error = False
    response.json.return_value = {"processing": {"status": "completed"}}
    client = IncidentRAGAPIClient("http://api:8000")

    with patch(
        "incidentrag.approval.ui.api_client.httpx.request", return_value=response
    ) as request:
        client.analyze_issue(42)

    assert request.call_args.kwargs["timeout"] == 300.0
