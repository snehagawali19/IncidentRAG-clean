"""Graph Layer — Public Facade."""
from __future__ import annotations

from incidentrag.graph.client import Neo4jClient
from incidentrag.graph.traversal import (
    BlastRadiusAnalyzer,
    EpisodicMemory,
    RelatedRunbooksRetriever,
)


class GraphLayer:
    """Single entry point for all graph operations."""

    def __init__(self) -> None:
        self.client = Neo4jClient()
        self.blast_radius = BlastRadiusAnalyzer(self.client)
        self.memory = EpisodicMemory(self.client)
        self.runbooks = RelatedRunbooksRetriever(self.client)

    async def setup(self) -> None:
        await self.client.ensure_schema()

    async def close(self) -> None:
        await self.client.close()
