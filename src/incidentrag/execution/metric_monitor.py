"""Post-execution metric verification."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricCheck:
    healthy: bool
    value: float | None
    attempts: int
    message: str


class MetricMonitor:
    """Poll a bounded health probe after an approved remediation."""

    def __init__(self, interval_seconds: float = 2.0, max_attempts: int = 5) -> None:
        self.interval_seconds = max(0.0, interval_seconds)
        self.max_attempts = max(1, max_attempts)

    async def verify(
        self,
        probe: Callable[[], Awaitable[tuple[bool, float | None]]],
    ) -> MetricCheck:
        last_value: float | None = None
        for attempt in range(1, self.max_attempts + 1):
            healthy, last_value = await probe()
            if healthy:
                return MetricCheck(True, last_value, attempt, "Metric recovered")
            if attempt < self.max_attempts:
                await asyncio.sleep(self.interval_seconds)
        return MetricCheck(False, last_value, self.max_attempts, "Metric did not recover")
