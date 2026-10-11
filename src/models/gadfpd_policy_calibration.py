"""Chronological calibration and fixed-volume alert policies for GADFPD.

This is deliberately research-only.  It reads the modern GitHub Actions
research dataset, never touches ``artifacts/``, and does not alter the API.
Thresholds are selected on a chronological validation period and evaluated only
on a later untouched test period.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

from src.inference.calibration import ProbabilityCalibrator
from src.models.chronological_calibration import expected_calibration_error
from src.models.gadfpd_research_evaluation import (
    DATA_PATH,
    FEATURES,
    RANDOM_STATE,
    TARGET,
    TIME,
    _preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "reports" / "gadfpd_calibration_policy_results.csv"
TARGET_ALERT_RATES = (0.05, 0.10, 0.15, 0.20)


def _random_forest() -> RandomForestClassifier:
    """Use the fixed project Random Forest configuration, without tuning."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def split_chronologically(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ordered data into earliest 72%, next 8%, and final 20%."""
    ordered = data.sort_values(TIME, kind="mergesort").reset_index(drop=True)
    model_end = int(len(ordered) * 0.72)
    test_start = int(len(ordered) * 0.80)
    model_train = ordered.iloc[:model_end].copy()
    calibration = ordered.iloc[model_end:test_start].copy()
    test = ordered.iloc[test_start:].copy()
    if not (len(model_train) and len(calibration) and len(test)):
        raise ValueError("GADFPD data is too small for a 72/8/20 chronological split.")
    if calibration[TIME].min() < model_train[TIME].max() or test[TIME].min() < calibration[TIME].max():
        raise RuntimeError("Chronological GADFPD split overlap detected.")
    return model_train, calibration, test


def threshold_at_alert_rate(probabilities: np.ndarray, alert_rate: float) -> float:
    """Return the validation-derived cutoff for an intended top-risk volume."""
    if not 0 < alert_rate < 1:
        raise ValueError("alert_rate must be strictly between zero and one.")
    values = np.sort(np.asarray(probabilities, dtype=float))
    if not len(values):
        raise ValueError("Cannot choose a threshold from no probabilities.")
    # ``>= threshold`` includes the score at the selected index.  With 100
    # distinct scores and a 15% policy, index 85 gives exactly 15 alerts;
    # index 84 would quietly yield 16.
    index = max(0, min(len(values) - 1, int(np.ceil((1 - alert_rate) * len(values)))))
    return float(values[index])


def _policy_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = np.asarray(probabilities >= threshold, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "TestAlertRate": float(predictions.mean()),
        "TestAlertCount": int(predictions.sum()),
        "Precision": float(precision),
        "FailureCaptureRate": float(recall),
        "F1": float(f1),
    }


def _read_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError("Run `python -m src.data.gadfpd_pipeline` before this evaluation.")
    data = pd.read_parquet(data_path, columns=[TIME, TARGET, *FEATURES])
    data[TIME] = pd.to_datetime(data[TIME], utc=True, errors="raise")
    return data.sort_values(TIME, kind="mergesort")


def run_gadfpd_policy_calibration(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Evaluate raw/sigmoid/isotonic probabilities under validation-set policies."""
    train, calibration, test = split_chronologically(_read_data(data_path))
    pipeline = Pipeline([("preprocessing", _preprocessor()), ("model", _random_forest())])
    started = time.perf_counter()
    pipeline.fit(train[FEATURES], train[TARGET])
    training_seconds = time.perf_counter() - started

    raw_validation = pipeline.predict_proba(calibration[FEATURES])[:, 1]
    raw_test = pipeline.predict_proba(test[FEATURES])[:, 1]
    y_validation, y_test = calibration[TARGET], test[TARGET]
    calibrators = {
        method: ProbabilityCalibrator.fit(method, raw_validation, y_validation)
        for method in ("raw", "sigmoid", "isotonic")
    }
    rows: list[dict] = []
    for method, calibrator in calibrators.items():
        validation_probabilities = calibrator.transform(raw_validation)
        test_probabilities = calibrator.transform(raw_test)
        shared = {
            "Method": method,
            "ModelTrainRows": len(train),
            "CalibrationRows": len(calibration),
            "FinalTestRows": len(test),
            "TrainingSeconds": float(training_seconds),
            "TestBrier": float(brier_score_loss(y_test, test_probabilities)),
            "TestECE": expected_calibration_error(y_test, test_probabilities),
            "TestAUC": float(roc_auc_score(y_test, test_probabilities)),
        }
        for alert_rate in TARGET_ALERT_RATES:
            threshold = threshold_at_alert_rate(validation_probabilities, alert_rate)
            rows.append(
                {
                    **shared,
                    "TargetAlertRate": alert_rate,
                    "ValidationSelectedThreshold": threshold,
                    **_policy_metrics(y_test, test_probabilities, threshold),
                }
            )
    results = pd.DataFrame(rows)
    # Isotonic calibration can assign a large plateau of identical scores.  A
    # real threshold must include every tied score, so TestAlertRate is reported
    # rather than pretending the requested volume was exactly attained.
    results["TargetVolumeDeviation"] = results["TestAlertRate"] - results["TargetAlertRate"]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    return results


def main() -> None:
    results = run_gadfpd_policy_calibration()
    print(results.to_string(index=False))
    print(f"Saved research-only results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
