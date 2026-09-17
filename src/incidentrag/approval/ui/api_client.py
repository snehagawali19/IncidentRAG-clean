"""Small typed HTTP client used by the Streamlit operator console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


def prefer_ipv4_loopback(url: str) -> str:
    """Use 127.0.0.1 so Windows clients do not stall on IPv6 localhost."""
    parts = urlsplit(url)
    hostname = (parts.hostname or "").lower()
    if hostname != "localhost":
        return url.rstrip("/")
    host = "127.0.0.1"
    if parts.port:
        host = f"{host}:{parts.port}"
    elif parts.scheme == "https":
        host = f"{host}:443"
    netloc = host
    if parts.username:
        user = parts.username
        if parts.password:
            user = f"{user}:{parts.password}"
        netloc = f"{user}@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")


class IncidentRAGAPIError(RuntimeError):
    """A safe, user-facing API failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.reason_phrase or "Unexpected backend response"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return response.reason_phrase or "Unexpected backend response"


@dataclass(frozen=True)
class IncidentRAGAPIClient:
    base_url: str
    api_key: str = ""
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", prefer_ipv4_loopback(self.base_url))

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        clean_params = {
            key: value
            for key, value in (params or {}).items()
            if value is not None and value != ""
        }
        try:
            response = httpx.request(
                method,
                f"{self.base_url.rstrip('/')}{path}",
                params=clean_params,
                headers=self.headers,
                timeout=timeout or self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise IncidentRAGAPIError(
                "The backend took too long to respond. Check its logs and try again."
            ) from exc
        except httpx.NetworkError as exc:
            raise IncidentRAGAPIError(
                "The frontend cannot reach the IncidentRAG backend."
            ) from exc
        if response.is_error:
            raise IncidentRAGAPIError(
                _error_detail(response), status_code=response.status_code
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise IncidentRAGAPIError("The backend returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise IncidentRAGAPIError("The backend returned an unexpected response.")
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def github_status(self) -> dict[str, Any]:
        return self._request("GET", "/sources/github/status")

    def find_issues(
        self, *, keyword: str, component: str, limit: int
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/sources/github/issues",
            params={"keyword": keyword, "component": component, "limit": limit},
        )

    def analyze_issue(self, issue_number: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/sources/github/issues/{issue_number}/analyze",
            timeout=300.0,
        )
