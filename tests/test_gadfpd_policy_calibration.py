"""Tests for chronology and validation-derived alert policy helpers."""

import numpy as np
import pandas as pd

from src.models.gadfpd_policy_calibration import split_chronologically, threshold_at_alert_rate


def test_chronological_split_is_ordered_and_exhaustive() -> None:
    data = pd.DataFrame(
        {
            "created_at": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"),
            "target_failed": [0, 1] * 50,
        }
    )
    train, calibration, test = split_chronologically(data)
    assert (len(train), len(calibration), len(test)) == (72, 8, 20)
    assert train["created_at"].max() <= calibration["created_at"].min()
    assert calibration["created_at"].max() <= test["created_at"].min()


def test_threshold_is_derived_from_requested_alert_volume() -> None:
    probabilities = np.linspace(0.01, 1.0, 100)
    threshold = threshold_at_alert_rate(probabilities, 0.15)
    assert 0.84 <= threshold <= 0.86
    assert float((probabilities >= threshold).mean()) == 0.15
