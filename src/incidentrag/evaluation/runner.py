"""
Phase 7 — Evaluation Harness.

RAGASRunner:
  - Runs RAGAS four canonical metrics (faithfulness, answer_relevancy,
    context_precision, context_recall) via the ragas library.
  - Adds three IncidentRAG-specific metrics:
      * rca_accuracy               — string similarity vs ground truth
      * remediation_safety_score   — 1.0 minus (dangerous_action_count / total_actions)
      * grounding_verification_rate — fraction of claims where verified=True

CIGate:
  - Enforces minimum faithfulness (from settings.eval_min_faithfulness).
  - Additional thresholds are configurable but default to 0.75 each.

EvaluationCaseBuilder:
  - Loads EvaluationCase objects from data/ground_truth/eval_set.jsonl.
"""
from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Any

from incidentrag.core.models import (
    EvaluationCase,
    IncidentAssessment,
    RAGASMetrics,
    RemediationAction,
    RiskLevel,
)
from incidentrag.core.settings import get_settings

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    logger.warning("ragas not installed — evaluation will use fallback scoring")


# Minimum thresholds — settings.eval_min_faithfulness overrides for faithfulness
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.75,
    "context_precision": 0.75,
    "context_recall": 0.70,
    "rca_accuracy": 0.60,
    "remediation_safety_score": 0.95,
    "grounding_verification_rate": 0.80,
}


def _string_similarity(a: str, b: str) -> float:
    """Simple sequence-matcher ratio for RCA accuracy."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _rca_accuracy(assessment: IncidentAssessment, ground_truth: str) -> float:
    """Compare root_cause.statement against ground truth string."""
    return _string_similarity(assessment.root_cause.statement, ground_truth)


def _remediation_safety(actions: list[RemediationAction], dangerous_to_avoid: list[str]) -> float:
    """
    Returns 1.0 if none of the dangerous_actions_to_avoid appear in commands.
    Returns 0.0 if any dangerous action is present.
    Otherwise a proportional score.
    """
    if not actions:
        return 1.0

    dangerous_hits = 0
    for action in actions:
        # Any action with DANGEROUS risk level counts against us
        if action.risk_level == RiskLevel.DANGEROUS:
            dangerous_hits += 1
            continue
        # Or any action containing a forbidden command substring
        cmd = (action.command or "").lower()
        for forbidden in dangerous_to_avoid:
            if forbidden.lower() in cmd:
                dangerous_hits += 1
                break

    return max(0.0, 1.0 - (dangerous_hits / len(actions)))


def _grounding_verification_rate(assessment: IncidentAssessment) -> float:
    claims = [assessment.root_cause, *assessment.contributing_factors]
    if not claims:
        return 0.0
    verified = sum(1 for c in claims if c.verified)
    return verified / len(claims)


class RAGASRunner:
    """
    Runs full evaluation over a set of (case, assessment) pairs.

    Each entry is a tuple: (case, assessment, generated_answer, retrieved_contexts)
      - case              : EvaluationCase — ground truth
      - assessment        : IncidentAssessment — model output
      - generated_answer  : str — LLM's answer text (usually root_cause.statement)
      - retrieved_contexts: list[str] — chunk contents fed into the LLM
    """

    async def run(
        self,
        pairs: list[tuple[EvaluationCase, IncidentAssessment, str, list[str]]],
    ) -> RAGASMetrics:
        if not pairs:
            return _empty_metrics()

        # ── RAGAS four canonical metrics ──────────────────────────────────
        if _RAGAS_AVAILABLE:
            try:
                dataset = Dataset.from_dict({
                    "question": [c.ground_truth_root_cause for c, _, _, _ in pairs],
                    "answer": [ans for _, _, ans, _ in pairs],
                    "contexts": [ctx for _, _, _, ctx in pairs],
                    "ground_truth": [c.ground_truth_root_cause for c, _, _, _ in pairs],
                })
                result = evaluate(
                    dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                )
                scores = result.to_pandas()
                faith = float(scores["faithfulness"].mean())
                relev = float(scores["answer_relevancy"].mean())
                prec = float(scores["context_precision"].mean())
                recall = float(scores["context_recall"].mean())
            except Exception as exc:  # noqa: BLE001
                logger.error("RAGAS eval failed, using fallback: %s", exc)
                faith, relev, prec, recall = _fallback_ragas(pairs)
        else:
            faith, relev, prec, recall = _fallback_ragas(pairs)

        # ── Custom IncidentRAG metrics ────────────────────────────────────
        rca_scores = [_rca_accuracy(a, c.ground_truth_root_cause) for c, a, _, _ in pairs]
        safety_scores = [
            _remediation_safety(a.proposed_actions, c.dangerous_actions_to_avoid)
            for c, a, _, _ in pairs
        ]
        grounding_scores = [_grounding_verification_rate(a) for _, a, _, _ in pairs]

        return RAGASMetrics(
            faithfulness=faith,
            answer_relevancy=relev,
            context_precision=prec,
            context_recall=recall,
            rca_accuracy=sum(rca_scores) / len(rca_scores),
            remediation_safety_score=sum(safety_scores) / len(safety_scores),
            grounding_verification_rate=sum(grounding_scores) / len(grounding_scores),
        )


def _fallback_ragas(
    pairs: list[tuple[EvaluationCase, IncidentAssessment, str, list[str]]],
) -> tuple[float, float, float, float]:
    """
    Fallback when ragas is not installed / fails: uses simple string similarity.
    """
    faith_vals, relev_vals, prec_vals, recall_vals = [], [], [], []
    for case, _assessment, answer, contexts in pairs:
        gt = case.ground_truth_root_cause
        # Answer vs ground truth
        relev_vals.append(_string_similarity(answer, gt))
        # Answer vs contexts (rough faithfulness)
        ctx_blob = " ".join(contexts)[:4000]
        faith_vals.append(_string_similarity(answer, ctx_blob))
        # Contexts vs ground truth (rough precision/recall)
        prec_vals.append(_string_similarity(ctx_blob[:1000], gt))
        recall_vals.append(_string_similarity(gt, ctx_blob[:2000]))

    def _avg(v: list[float]) -> float:
        return sum(v) / len(v) if v else 0.0

    return _avg(faith_vals), _avg(relev_vals), _avg(prec_vals), _avg(recall_vals)


def _empty_metrics() -> RAGASMetrics:
    return RAGASMetrics(
        faithfulness=0.0,
        answer_relevancy=0.0,
        context_precision=0.0,
        context_recall=0.0,
        rca_accuracy=0.0,
        remediation_safety_score=0.0,
        grounding_verification_rate=0.0,
    )


class CIGate:
    """Enforces evaluation thresholds — raises RuntimeError on failure."""

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        s = get_settings()
        base = dict(DEFAULT_THRESHOLDS)
        base["faithfulness"] = s.eval_min_faithfulness
        if thresholds:
            base.update(thresholds)
        self._thresholds = base

    def check(self, metrics: RAGASMetrics) -> None:
        failures: list[str] = []
        for name, threshold in self._thresholds.items():
            value = getattr(metrics, name, None)
            if value is None:
                continue
            if value < threshold:
                failures.append(f"{name}={value:.3f} < {threshold}")

        if failures:
            raise RuntimeError("CI gate failed:\n" + "\n".join(f"  - {f}" for f in failures))
        logger.info("CI gate passed — all metrics above thresholds")


class EvaluationCaseBuilder:
    """Loads / persists EvaluationCase objects."""

    def __init__(self, path: Path | None = None) -> None:
        s = get_settings()
        self._path = path or Path(s.eval_dataset_path)

    def load(self) -> list[EvaluationCase]:
        if not self._path.exists():
            logger.warning("Eval set not found at %s", self._path)
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

    def append(self, case: EvaluationCase) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(case.model_dump_json() + "\n")
