"""Tests for Neo4jClient, BlastRadiusAnalyzer, EpisodicMemory, RelatedRunbooksRetriever."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from incidentrag.core.models import BlastRadiusResult, ServiceNode
from incidentrag.graph.client import Neo4jClient, _CONSTRAINTS
from incidentrag.graph.traversal import (
    BlastRadiusAnalyzer,
    EpisodicMemory,
    PastIncident,
    RelatedRunbooksRetriever,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_neo4j_client() -> AsyncMock:
    """A mock Neo4jClient with run returning [] by default."""
    client = AsyncMock(spec=Neo4jClient)
    client.run = AsyncMock(return_value=[])
    return client


@pytest.fixture
def blast_analyzer(mock_neo4j_client) -> BlastRadiusAnalyzer:
    return BlastRadiusAnalyzer(mock_neo4j_client)


@pytest.fixture
def episodic_memory(mock_neo4j_client) -> EpisodicMemory:
    return EpisodicMemory(mock_neo4j_client)


@pytest.fixture
def runbook_retriever(mock_neo4j_client) -> RelatedRunbooksRetriever:
    return RelatedRunbooksRetriever(mock_neo4j_client)


# ── Neo4jClient tests ─────────────────────────────────────────────────────────

async def test_neo4j_client_runs_cypher():
    """Neo4jClient.run() should call session.run with the provided Cypher."""
    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=[{"n": "value"}])
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session_cm)

    with patch("incidentrag.graph.client.AsyncGraphDatabase") as mock_db:
        mock_db.driver.return_value = mock_driver
        client = Neo4jClient()
        cypher = "MATCH (n:Service) RETURN n"
        await client.run(cypher)
        mock_session.run.assert_called_once_with(cypher)


async def test_ensure_schema_runs_constraints():
    """ensure_schema() should run exactly 3 CREATE CONSTRAINT statements."""
    mock_session = AsyncMock()
    mock_session.run = AsyncMock()

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session_cm)

    with patch("incidentrag.graph.client.AsyncGraphDatabase") as mock_db:
        mock_db.driver.return_value = mock_driver
        client = Neo4jClient()
        await client.ensure_schema()
        assert mock_session.run.call_count == len(_CONSTRAINTS)


# ── BlastRadiusAnalyzer tests ─────────────────────────────────────────────────

async def test_blast_radius_empty_graph(blast_analyzer, mock_neo4j_client):
    """When the graph has no dependencies, BlastRadiusResult is returned with hop_depth=0."""
    mock_neo4j_client.run = AsyncMock(return_value=[])
    result = await blast_analyzer.analyze("payment-service")
    assert isinstance(result, BlastRadiusResult)
    assert result.hop_depth == 0
    assert result.affected_services == []


async def test_blast_radius_single_hop(blast_analyzer, mock_neo4j_client):
    """With one dependency, affected_services should have 1 ServiceNode."""
    mock_neo4j_client.run = AsyncMock(
        return_value=[
            {"dep_name": "svc-b", "hops": 1, "path_nodes": ["payment-service", "svc-b"]}
        ]
    )
    result = await blast_analyzer.analyze("payment-service")
    assert len(result.affected_services) == 1
    assert isinstance(result.affected_services[0], ServiceNode)
    assert result.affected_services[0].service_name == "svc-b"


async def test_blast_radius_origin_service(blast_analyzer, mock_neo4j_client):
    """result.origin_service should match the service passed to analyze()."""
    mock_neo4j_client.run = AsyncMock(return_value=[])
    result = await blast_analyzer.analyze("payment-service")
    assert result.origin_service == "payment-service"


async def test_blast_radius_embeds_bounded_hop_count(
    blast_analyzer,
    mock_neo4j_client,
):
    """Neo4j relationship bounds must be safe query literals, not parameters."""
    await blast_analyzer.analyze("payment-service", max_hops=99)

    query = mock_neo4j_client.run.await_args.args[0]
    assert "[:DEPENDS_ON*1..5]" in query
    assert "$hops" not in query
    assert mock_neo4j_client.run.await_args.kwargs == {"service": "payment-service"}


# ── EpisodicMemory tests ──────────────────────────────────────────────────────

async def test_episodic_memory_get_similar_empty(episodic_memory, mock_neo4j_client):
    """get_similar() with no DB results should return an empty list."""
    mock_neo4j_client.run = AsyncMock(return_value=[])
    result = await episodic_memory.get_similar("payment-service")
    assert result == []


async def test_episodic_memory_get_similar_formats_past_incidents(episodic_memory, mock_neo4j_client):
    """get_similar() should map DB records to PastIncident objects."""
    mock_neo4j_client.run = AsyncMock(return_value=[
        {
            "id": "inc-123",
            "title": "OOMKilled payment pod",
            "summary": "Increased memory limit to 1Gi",
            "mttr": 15.0,
            "runbooks": ["rb-payment-memory"],
        }
    ])
    result = await episodic_memory.get_similar("payment-service")
    assert len(result) == 1
    assert isinstance(result[0], PastIncident)
    assert result[0].id == "inc-123"
    assert result[0].mttr_minutes == 15.0


# ── RelatedRunbooksRetriever tests ────────────────────────────────────────────

async def test_related_runbooks_empty_services(runbook_retriever, mock_neo4j_client):
    """get_runbook_ids([]) should return [] without calling the DB."""
    result = await runbook_retriever.get_runbook_ids([])
    assert result == []
    mock_neo4j_client.run.assert_not_called()


async def test_related_runbooks_queries_db(runbook_retriever, mock_neo4j_client):
    """With a services list, get_runbook_ids() should query the DB and return runbook IDs."""
    mock_neo4j_client.run = AsyncMock(return_value=[
        {"runbook_id": "rb-payment-memory"},
        {"runbook_id": "rb-kubernetes-crashloop"},
    ])
    result = await runbook_retriever.get_runbook_ids(["payment-service"])
    mock_neo4j_client.run.assert_called_once()
    assert "rb-payment-memory" in result
    assert "rb-kubernetes-crashloop" in result
