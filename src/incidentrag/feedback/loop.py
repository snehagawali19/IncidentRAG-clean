"""
Phase 10 — Feedback Loop.

Two components:

1. ChunkScoreUpdater
   - On successful incident resolution, boost chunk.boost_score by +0.05 (capped at 2.0)
   - For every chunk_id in IncidentAssessment.evidence_chunks_used
   - Persisted back to the Qdrant payload (via a DenseIndex handle)

2. EvalSetExpander
   - Converts a resolved IncidentAssessment into an EvaluationCase
   - Appends to the persistent eval set (jsonl at settings.eval_dataset_path)
   - Enables continuous improvement of the RAGAS harness
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Protocol

from incidentrag.core.models import (
    EvaluationCase,
    IncidentAssessment,
    RawAlert,
    RiskLevel,
)
from incidentrag.core.settings import get_settings

logger = logging.getLogger(__name__)

BOOST_INCREMENT = 0.05
BOOST_CAP = 2.0


class _ChunkStoreProtocol(Protocol):
    """Minimal interface required from a chunk store to apply score boosts."""

    async def get_boost_score(self, chunk_id: str) -> float | None: ...
    async def set_boost_score(self, chunk_id: str, boost_score: float) -> None: ...


class ChunkScoreUpdater:
    """Applies feedback boosts to chunks used in successful resolutions."""

    def __init__(self, chunk_store: _ChunkStoreProtocol) -> None:
        self._store = chunk_store

    async def boost_used_chunks(self, assessment: IncidentAssessment) -> dict[str, float]:
        """
        For every chunk_id in evidence_chunks_used, bump its boost_score by
        BOOST_INCREMENT, capped at BOOST_CAP.

        Returns a dict of {chunk_id: new_boost_score}.
        """
        updated: dict[str, float] = {}

        for chunk_id in assessment.evidence_chunks_used:
            if chunk_id == "none":
                continue
            try:
                current = await self._store.get_boost_score(chunk_id)
                if current is None:
                    current = 1.0
                new_boost = min(current + BOOST_INCREMENT, BOOST_CAP)
                await self._store.set_boost_score(chunk_id, new_boost)
                updated[chunk_id] = new_boost
                logger.debug("Boosted chunk %s: %.3f → %.3f", chunk_id, current, new_boost)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to boost chunk %s: %s", chunk_id, exc)

        logger.info(
            "Feedback boost applied to %d chunks for incident %s",
            len(updated), assessment.incident_id,
        )
        return updated


class EvalSetExpander:
    """Adds resolved incidents to the persistent eval set as new EvaluationCase objects."""

    def __init__(self, eval_set_path: Path | None = None) -> None:
        s = get_settings()
        self._path = eval_set_path or Path(s.eval_dataset_path)

    def _ensure_path(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def expand(
        self,
        assessment: IncidentAssessment,
        original_alert: RawAlert,
        ground_truth_root_cause: str | None = None,
        ground_truth_relevant_runbooks: list[str] | None = None,
        ground_truth_correct_actions: list[str] | None = None,
        dangerous_actions_to_avoid: list[str] | None = None,
        incident_source: str | None = None,
    ) -> EvaluationCase:
        """
        Create and persist an EvaluationCase from a resolved incident.

        If ground truth is not supplied, uses the model's own outputs as
        soft ground truth (still useful for regression testing).
        """
        proposed_cmds = [
            a.command or a.description
            for a in assessment.proposed_actions
        ]
        auto_dangerous = [
            a.command or a.description
            for a in assessment.proposed_actions
            if a.risk_level == RiskLevel.DANGEROUS
        ]

        case = EvaluationCase(
            case_id=f"case-{uuid.uuid4()}",
            incident_source=incident_source or f"resolved-{assessment.incident_id}",
            input_alerts=[original_alert],
            ground_truth_root_cause=(
                ground_truth_root_cause or assessment.root_cause.statement
            ),
            ground_truth_relevant_runbooks=(
                ground_truth_relevant_runbooks
                or list({ev.runbook_id for ev in assessment.root_cause.evidence})
            ),
            ground_truth_correct_actions=(
                ground_truth_correct_actions or proposed_cmds
            ),
            dangerous_actions_to_avoid=(
                dangerous_actions_to_avoid or auto_dangerous
            ),
        )

        self._ensure_path()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(case.model_dump_json() + "\n")

        logger.info(
            "Eval set expanded with incident %s → %s",
            assessment.incident_id, self._path,
        )
        return case

    def load(self) -> list[EvaluationCase]:
        if not self._path.exists():
            return []
        cases: list[EvaluationCase] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(EvaluationCase.model_validate_json(line))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping malformed eval case: %s", exc)
        return cases


# ── Simple in-memory chunk store for testing ──────────────────────────────

class InMemoryChunkStore:
    """A trivial chunk store that satisfies _ChunkStoreProtocol for tests."""

    def __init__(self, initial: dict[str, float] | None = None) -> None:
        self._scores: dict[str, float] = dict(initial or {})

    async def get_boost_score(self, chunk_id: str) -> float | None:
        return self._scores.get(chunk_id)

    async def set_boost_score(self, chunk_id: str, boost_score: float) -> None:
        self._scores[chunk_id] = boost_score

    def snapshot(self) -> dict[str, float]:
        return dict(self._scores)
