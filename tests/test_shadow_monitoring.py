"""Tests for pure metric helpers used by the read-only shadow monitor."""

import pandas as pd

from src.models.shadow_monitoring import _grouped_metrics, _metrics


def test_shadow_metrics_measure_alert_volume_precision_and_capture() -> None:
    frame = pd.DataFrame(
        {
            "actual_failed": [1, 1, 0, 0],
            "alert": [1, 0, 1, 0],
            "pipeline": ["one", "one", "two", "two"],
        }
    )
    metrics = _metrics(frame)
    assert metrics == {
        "resolved_builds": 4,
        "actual_failure_rate": 0.5,
        "alert_rate": 0.5,
        "tier_precision": 0.5,
        "failure_capture_rate": 0.5,
    }
    grouped = _grouped_metrics(frame, "pipeline")
    assert set(grouped["pipeline"]) == {"one", "two"}
