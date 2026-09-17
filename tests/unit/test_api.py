"""Tests for the FastAPI application in incidentrag.api.app."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.unit.conftest import make_assessment


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_client():
    """
    Create a TestClient with DenseIndex and GraphLayer constructors patched
    so no real Qdrant / Neo4j connections are attempted.
    """
    with patch("incidentrag.retrieval.dense_index.AsyncQdrantClient") as mock_qdrant_cls, \
         patch("incidentrag.retrieval.dense_index.AsyncOpenAI") as mock_oai_cls, \
         patch("incidentrag.graph.client.AsyncGraphDatabase") as mock_neo4j_cls:

        # Qdrant mock — ensure_collection and search return gracefully
        qdrant_inst = AsyncMock()
        qdrant_inst.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )
        qdrant_inst.create_collection = AsyncMock()
        mock_qdrant_cls.return_value = qdrant_inst

        # OpenAI mock
        oai_inst = AsyncMock()
        mock_oai_cls.return_value = oai_inst

        # Neo4j mock
        mock_driver = MagicMock()
        mock_session_cm = MagicMock()
        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)
        mock_driver.session = MagicMock(return_value=mock_session_cm)
        mock_driver.close = AsyncMock()
        mock_neo4j_cls.driver.return_value = mock_driver

        # Import app after patching so constructors use mocks
        from incidentrag.api import app as app_module
        # Reset assessments store
        app_module._assessments.clear()

        client = TestClient(app_module.app, raise_server_exceptions=False)
        return client, app_module


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    """GET /health should return 200 with status 'ok'."""
    client, _ = _make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_health_service_name():
    """GET /health should include service: 'incidentrag'."""
    client, _ = _make_client()
    resp = client.get("/health")
    assert resp.json()["service"] == "incidentrag"


def test_get_assessment_not_found():
    """GET /assessments/missing should return 404."""
    client, _ = _make_client()
    resp = client.get("/assessments/missing-incident-id")
    assert resp.status_code == 404


def test_get_assessment_found():
    """Injecting an assessment into _assessments → GET /assessments/{id} returns 200."""
    client, app_module = _make_client()
    assessment = make_assessment(incident_id="inc-test-123")
    app_module._assessments["inc-test-123"] = assessment
    resp = client.get("/assessments/inc-test-123")
    assert resp.status_code == 200
    data = resp.json()
    assert data["incident_id"] == "inc-test-123"


def test_list_approvals_empty():
    """GET /approvals should return 200 with an empty list when no requests pending."""
    client, _ = _make_client()
    resp = client.get("/approvals")
    assert resp.status_code == 200
    assert resp.json() == []


def test_approve_unknown_request():
    """POST /approvals/missing/approve should return 400 or 503."""
    client, _ = _make_client()
    resp = client.post("/approvals/nonexistent-request-id/approve")
    assert resp.status_code in (400, 503)


def test_reject_unknown_request():
    """POST /approvals/missing/reject should return 404 or 400."""
    client, _ = _make_client()
    resp = client.post("/approvals/nonexistent-request-id/reject")
    assert resp.status_code in (400, 404, 503)
