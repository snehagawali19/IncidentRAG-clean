"""Observability Layer — Embedding Drift Detector.

Monitors per-class (incident category) centroid drift.
Uses z-score alerting: |z| > 3.0 triggers a drift warning.

Adapted from the 50-line embedding drift monitor pattern
described in dev.to/embspec.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone

from incidentrag.core.models import DriftMeasurement

logger = logging.getLogger(__name__)

_Z_THRESHOLD = 3.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


class DriftDetector:
    """
    Maintains a rolling window of embeddings per class label.
    On each update, computes similarity to the class centroid
    and flags if the z-score exceeds the threshold.
    """

    def __init__(self, window_size: int = 100, z_threshold: float = _Z_THRESHOLD) -> None:
        self._window_size = window_size
        self._z_threshold = z_threshold
        # class_label -> list of embeddings (as tuples for hashability)
        self._embeddings: dict[str, list[list[float]]] = defaultdict(list)
        self._centroids: dict[str, list[float]] = {}

    def _compute_centroid(self, embeddings: list[list[float]]) -> list[float]:
        if not embeddings:
            return []
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                centroid[i] += v
        n = len(embeddings)
        return [v / n for v in centroid]

    def update(self, class_label: str, embedding: list[float]) -> DriftMeasurement | None:
        """
        Add a new embedding for a class. Returns a DriftMeasurement
        if enough data exists, else None.
        """
        window = self._embeddings[class_label]
        window.append(embedding)

        # Maintain rolling window
        if len(window) > self._window_size:
            window.pop(0)

        if len(window) < 10:
            return None  # not enough data yet

        centroid = self._compute_centroid(window[:-1])  # exclude newest
        self._centroids[class_label] = centroid

        # Compute similarities to centroid for all historical embeddings
        sims = [_cosine_similarity(e, centroid) for e in window[:-1]]
        mean_sim = _mean(sims)
        std_sim = _std(sims, mean_sim)

        # Z-score of the newest embedding vs historical distribution
        newest_sim = _cosine_similarity(embedding, centroid)
        z = (newest_sim - mean_sim) / std_sim if std_sim > 0 else 0.0

        drift = abs(z) > self._z_threshold
        if drift:
            logger.warning(
                "Embedding drift detected for class '%s': z=%.2f (threshold=%.1f)",
                class_label, z, self._z_threshold,
            )

        return DriftMeasurement(
            class_label=class_label,
            mean_similarity=mean_sim,
            std_similarity=std_sim,
            noise_floor_max=0.0,          # not computed in this simplified impl
            signal_gap=newest_sim - 0.0,  # distance above noise floor
            z_score_vs_baseline=z,
            is_drift_detected=drift,
        )

    def get_centroid(self, class_label: str) -> list[float] | None:
        return self._centroids.get(class_label)
