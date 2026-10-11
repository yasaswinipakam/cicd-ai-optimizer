"""Robust timing-based CI job bottleneck and in-run stall assessment.

This module consumes runtime telemetry only. It is deliberately isolated from
the pre-build failure classifier, because durations do not exist at prediction
time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_HISTORY = 10


@dataclass(frozen=True)
class TimingAssessment:
    status: str
    sample_size: int
    median_seconds: float | None
    p95_seconds: float | None
    threshold_seconds: float | None


def assess_job_timing(current_seconds: float, historical_seconds, min_history: int = MIN_HISTORY) -> TimingAssessment:
    """Assess a job against prior matching completed jobs.

    A job is anomalously slow only if there are enough historical observations
    and it exceeds both p95 and a robust median-plus-3-MAD threshold. The same
    logic can flag an in-progress job as ``stalled`` based on elapsed runtime.
    """
    history = np.asarray([value for value in historical_seconds if value is not None and value >= 0], dtype=float)
    if len(history) < min_history:
        return TimingAssessment("insufficient_history", int(len(history)), None, None, None)
    median = float(np.median(history))
    p95 = float(np.percentile(history, 95))
    mad = float(np.median(np.abs(history - median)))
    robust_threshold = median + 3 * 1.4826 * mad
    threshold = float(max(p95, robust_threshold))
    status = "bottleneck" if current_seconds > threshold else "normal"
    return TimingAssessment(status, int(len(history)), median, p95, threshold)
