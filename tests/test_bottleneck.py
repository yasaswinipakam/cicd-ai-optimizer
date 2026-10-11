"""Tests for runtime-only bottleneck/stall assessment."""

from src.models.bottleneck import assess_job_timing


def test_timing_assessment_requires_history_then_flags_slow_job() -> None:
    assert assess_job_timing(100, [10, 11]).status == "insufficient_history"
    history = [10, 11, 10, 12, 11, 10, 12, 11, 10, 11]
    normal = assess_job_timing(12, history)
    slow = assess_job_timing(100, history)
    assert normal.status == "normal"
    assert slow.status == "bottleneck"
    assert slow.sample_size == 10
    assert slow.threshold_seconds is not None and slow.threshold_seconds < 100
