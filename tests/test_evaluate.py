"""Artifact-level checks for reproducible evaluation outputs."""

from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "artifacts" / "model_results.csv"


def test_model_results_has_four_models_and_bounded_metrics() -> None:
    if not RESULTS_PATH.exists():
        pytest.skip("Evaluation artifacts have not been generated yet.")

    results = pd.read_csv(RESULTS_PATH)
    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    assert len(results) == 4
    assert list(results.columns) == ["Model", *metric_columns]
    assert results[metric_columns].apply(lambda column: column.between(0, 1).all()).all()
