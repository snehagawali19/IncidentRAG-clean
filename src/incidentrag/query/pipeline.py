"""Query Layer — Orchestration Pipeline.

Runs all Layer 2 components in the correct order:
  1. EntityExtractor   → ExtractedEntities
  2. IncidentClassifier → IncidentCategory (updates entities)
  3. QueryFanoutGenerator → list[str] (5 queries)
  4. HyDEGenerator     → str (hypothetical doc)
"""
from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from incidentrag.core.ai_provider import openai_client_kwargs
from incidentrag.core.models import ExtractedEntities, IncidentCategory, RawAlert
from incidentrag.query.classifier import IncidentClassifier
from incidentrag.query.entity_extractor import EntityExtractor
from incidentrag.query.fanout import QueryFanoutGenerator
from incidentrag.query.hyde import HyDEGenerator

logger = logging.getLogger(__name__)


class QueryUnderstandingPipeline:
    """Single entry point for Layer 2 processing."""

    def __init__(self) -> None:
        client = AsyncOpenAI(**openai_client_kwargs())
        self.extractor = EntityExtractor(client)
        self.classifier = IncidentClassifier(client)
        self.fanout = QueryFanoutGenerator(client)
        self.hyde = HyDEGenerator(client)

    async def process(
        self, alert: RawAlert
    ) -> tuple[ExtractedEntities, IncidentCategory, list[str], str]:
        """
        Returns:
            entities      — structured alert metadata (canonical ExtractedEntities)
            category      — classified IncidentCategory
            queries       — 5 semantic search queries
            hyde_document — hypothetical runbook section (300-500 words)
        """
        # Step 1: entity extraction
        entities = await self.extractor.extract(alert)

        # Step 2: classification — returned separately; not stored on entities
        # (ExtractedEntities has no incident_category field per models.py)
        category = await self.classifier.classify(alert, entities)

        # Step 3: fanout + HyDE in parallel
        queries, hyde_doc = await asyncio.gather(
            self.fanout.generate(alert, entities),
            self.hyde.generate(alert, entities),
        )

        logger.info(
            "Query understanding complete — service=%s category=%s queries=%d hyde_words=%d",
            entities.service,
            category.value,
            len(queries),
            len(hyde_doc.split()),
        )

        return entities, category, queries, hyde_doc
