"""Graph Layer — Neo4j Client + Schema Setup."""
from __future__ import annotations

import logging

from neo4j import AsyncGraphDatabase, AsyncDriver

from incidentrag.core.settings import settings

logger = logging.getLogger(__name__)

_CONSTRAINTS = [
    "CREATE CONSTRAINT service_name IF NOT EXISTS FOR (s:Service) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT incident_id IF NOT EXISTS FOR (i:Incident) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT runbook_id IF NOT EXISTS FOR (r:Runbook) REQUIRE r.id IS UNIQUE",
]


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def ensure_schema(self) -> None:
        async with self._driver.session() as session:
            for stmt in _CONSTRAINTS:
                try:
                    await session.run(stmt)
                except Exception as exc:
                    logger.debug("Schema stmt skipped: %s", exc)
        logger.info("Neo4j schema ensured")

    async def run(self, cypher: str, **params) -> list[dict]:  # type: ignore[type-arg]
        async with self._driver.session() as session:
            result = await session.run(cypher, **params)
            return await result.data()

    async def upsert_service(
        self,
        name: str,
        team: str = "",
        tier: int = 2,
        namespace: str = "",
        region: str = "",
    ) -> None:
        await self.run(
            """
            MERGE (s:Service {name: $name})
            SET s.team = $team, s.tier = $tier,
                s.namespace = $namespace, s.region = $region
            """,
            name=name, team=team, tier=tier, namespace=namespace, region=region,
        )

    async def add_dependency(self, src: str, dst: str, weight: float = 1.0) -> None:
        await self.run(
            """
            MERGE (a:Service {name: $src})
            MERGE (b:Service {name: $dst})
            MERGE (a)-[r:DEPENDS_ON]->(b)
            SET r.weight = $weight
            """,
            src=src, dst=dst, weight=weight,
        )
