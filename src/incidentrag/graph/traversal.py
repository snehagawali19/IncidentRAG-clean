"""Graph Layer — Blast Radius Analyzer + Episodic Memory + Related Runbooks."""
from __future__ import annotations

import logging

from incidentrag.core.models import BlastRadiusResult, ServiceNode
from incidentrag.graph.client import Neo4jClient

logger = logging.getLogger(__name__)


class BlastRadiusAnalyzer:
    """Multi-hop DEPENDS_ON traversal to compute blast radius."""

    def __init__(self, client: Neo4jClient) -> None:
        self._db = client

    async def analyze(self, service: str, max_hops: int = 3) -> BlastRadiusResult:
        safe_hops = max(1, min(int(max_hops), 5))
        records = await self._db.run(
            f"""
            MATCH path = (root:Service {{name: $service}})
              -[:DEPENDS_ON*1..{safe_hops}]->(dep:Service)
            RETURN dep.name AS dep_name,
                   length(path) AS hops,
                   [n IN nodes(path) | n.name] AS path_nodes
            ORDER BY hops
            """,
            service=service,
        )

        affected = list({r["dep_name"] for r in records})
        max_hop = max((r["hops"] for r in records), default=0)

        service_nodes = [
            ServiceNode(
                service_name=name,
                tier="api",
                environment="production",
                criticality="high",
            )
            for name in affected
        ]
        return BlastRadiusResult(
            origin_service=service,
            hop_depth=max_hop,
            affected_services=service_nodes,
            critical_path_services=affected[:10],
        )


class EpisodicMemory:
    """Retrieves historically similar incidents from Neo4j."""

    def __init__(self, client: Neo4jClient) -> None:
        self._db = client

    async def record_incident(
        self,
        incident_id: str,
        title: str,
        service: str,
        resolution_summary: str,
        mttr_minutes: float,
        runbook_ids: list[str],
    ) -> None:
        await self._db.run(
            """
            MERGE (i:Incident {id: $id})
            SET i.title = $title, i.resolution_summary = $summary, i.mttr_minutes = $mttr
            WITH i
            MATCH (s:Service {name: $service})
            MERGE (i)-[:AFFECTED]->(s)
            """,
            id=incident_id, title=title, summary=resolution_summary,
            mttr=mttr_minutes, service=service,
        )
        for rid in runbook_ids:
            await self._db.run(
                """
                MERGE (r:Runbook {id: $rid})
                WITH r
                MATCH (i:Incident {id: $iid})
                MERGE (i)-[:RESOLVED_BY]->(r)
                """,
                rid=rid, iid=incident_id,
            )

    async def get_similar(
        self, service: str, limit: int = 5
    ) -> list[PastIncident]:
        records = await self._db.run(
            """
            MATCH (i:Incident)-[:AFFECTED]->(s:Service {name: $service})
            OPTIONAL MATCH (i)-[:RESOLVED_BY]->(r:Runbook)
            RETURN i.id AS id, i.title AS title,
                   i.resolution_summary AS summary,
                   i.mttr_minutes AS mttr,
                   collect(r.id) AS runbooks
            ORDER BY i.mttr_minutes ASC
            LIMIT $limit
            """,
            service=service, limit=limit,
        )
        return [
            PastIncident(
                id=r["id"],
                title=r["title"],
                service=service,
                resolution_summary=r["summary"] or "",
                runbook_ids=r["runbooks"] or [],
                mttr_minutes=r["mttr"],
            )
            for r in records
        ]


class RelatedRunbooksRetriever:
    def __init__(self, client: Neo4jClient) -> None:
        self._db = client

    async def get_runbook_ids(self, services: list[str]) -> list[str]:
        if not services:
            return []
        records = await self._db.run(
            """
            MATCH (i:Incident)-[:AFFECTED]->(s:Service)
            WHERE s.name IN $services
            MATCH (i)-[:RESOLVED_BY]->(r:Runbook)
            RETURN DISTINCT r.id AS runbook_id
            LIMIT 20
            """,
            services=services,
        )
        return [r["runbook_id"] for r in records]


# ── Convenience model missing from core/models.py ─────────────────────────────
from pydantic import BaseModel, Field  # noqa: E402


class PastIncident(BaseModel):
    id: str
    title: str
    service: str
    resolution_summary: str
    runbook_ids: list[str] = Field(default_factory=list)
    mttr_minutes: float | None = None
