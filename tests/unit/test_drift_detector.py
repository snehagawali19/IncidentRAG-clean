"""Tests for Layer 8 — Embedding Drift Detector."""
from __future__ import annotations

import math
import random

import pytest

from incidentrag.observability.drift_detector import DriftDetector, _cosine_similarity


def _unit_vec(dim: int = 8, seed: int = 0) -> list[float]:
    random.seed(seed)
    v = [random.gauss(0, 1) for _ in range(dim)]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


def test_cosine_similarity_identical() -> None:
    v = _unit_vec()
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal() -> None:
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert abs(_cosine_similarity(v1, v2)) < 1e-9


def test_detector_returns_none_before_threshold() -> None:
    det = DriftDetector(window_size=100)
    for i in range(9):
        result = det.update("oom_kill", _unit_vec(seed=i))
    assert result is None  # needs 10 data points minimum


def test_detector_returns_measurement_after_10() -> None:
    det = DriftDetector()
    measurement = None
    base = _unit_vec(seed=42)
    for i in range(11):
        measurement = det.update("oom_kill", base)  # same vector each time
    assert measurement is not None
    assert measurement.class_label == "oom_kill"
    assert 0.0 <= measurement.mean_similarity <= 1.0


def test_detector_flags_drift_on_outlier() -> None:
    det = DriftDetector(z_threshold=2.0)
    base = _unit_vec(8, seed=1)

    # Build up a stable distribution
    for _ in range(50):
        noisy = [b + random.gauss(0, 0.001) for b in base]
        det.update("test_class", noisy)

    # Inject a completely different vector (orthogonal direction)
    outlier = _unit_vec(8, seed=999)
    result = det.update("test_class", outlier)

    assert result is not None
    assert result.is_drift_detected is True


def test_detector_no_false_positive_stable_distribution() -> None:
    """Stable embeddings should not trigger drift."""
    det = DriftDetector(z_threshold=3.0)
    base = _unit_vec(8, seed=5)
    result = None
    for _ in range(30):
        noisy = [b + random.gauss(0, 0.002) for b in base]
        result = det.update("stable", noisy)

    if result:  # might be None for early iterations
        assert result.is_drift_detected is False
