"""Tests for non-production GADFPD evaluation helpers."""

import pandas as pd

from src.models.gadfpd_research_evaluation import _select_repository_holdouts


def test_repository_selection_uses_later_period_size_and_failure_rules() -> None:
    cutoff = pd.Timestamp("2026-01-01", tz="UTC")
    data = pd.DataFrame(
        {
            "repo": ["large"] * 220 + ["small"] * 199,
            "created_at": [pd.Timestamp("2026-01-02", tz="UTC")] * 419,
            "target_failed": [1] * 20 + [0] * 200 + [1] * 30 + [0] * 169,
        }
    )
    assert _select_repository_holdouts(data, cutoff) == ["large"]
