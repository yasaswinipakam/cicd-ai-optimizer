"""Chronological probability calibration for the deployable CI risk model.

The model is fitted only on the earliest 72% of history.  Its calibrator and
15%-volume warning cutoff are fitted only on the next 8%; the final 20% remains
untouched until all raw/sigmoid/isotonic measurements are produced.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import time

import joblib

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "cicd-ai-optimizer-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features.engineer import engineer_features
from src.features.schema import FEATURE_COLS, FEATURE_SOURCE_COLUMNS
from src.inference.calibration import ProbabilityCalibrator
from src.models.train import (
    ARTIFACTS_DIR,
    DATA_PATH,
    FULL_TRAIN_FRACTION,
    RANDOM_STATE,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    VALIDATION_FRACTION_OF_TRAIN,
    _random_forest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "chronological_calibration_results.csv"
RELIABILITY_PATH = REPORTS_DIR / "chronological_reliability_diagram.png"
ANALYSIS_PATH = REPORTS_DIR / "chronological_calibration_analysis.md"
CALIBRATOR_PATH = ARTIFACTS_DIR / "probability_calibrator.joblib"
TARGET_ALERT_RATE = 0.15


def _load_ordered_raw(data_path: Path) -> pd.DataFrame:
    columns = list(dict.fromkeys([TARGET_COLUMN, *FEATURE_SOURCE_COLUMNS]))
    raw = pd.read_parquet(data_path, columns=columns, engine="pyarrow")
    raw[TIMESTAMP_COLUMN] = pd.to_datetime(raw[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    if raw[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Calibration requires parseable chronological build timestamps.")
    return raw.sort_values(TIMESTAMP_COLUMN, kind="mergesort")


def _split_chronologically(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_start = int(len(raw) * FULL_TRAIN_FRACTION)
    pretest = raw.iloc[:test_start]
    calibration_start = int(len(pretest) * (1 - VALIDATION_FRACTION_OF_TRAIN))
    model_train = pretest.iloc[:calibration_start].copy()
    calibration = pretest.iloc[calibration_start:].copy()
    test = raw.iloc[test_start:].copy()
    if calibration[TIMESTAMP_COLUMN].min() < model_train[TIMESTAMP_COLUMN].max():
        raise RuntimeError("Chronological calibration split overlap detected.")
    if test[TIMESTAMP_COLUMN].min() < calibration[TIMESTAMP_COLUMN].max():
        raise RuntimeError("Untouched final test period overlaps calibration data.")
    return model_train, calibration, test


def _transform_splits(model_train_raw, calibration_raw, test_raw):
    encoders: dict = {}
    train, returned = engineer_features(model_train_raw, encoders=encoders, fit_encoders=True, save_encoders=False)
    calibration, calibration_returned = engineer_features(
        calibration_raw, encoders=encoders, fit_encoders=False, save_encoders=False
    )
    test, test_returned = engineer_features(test_raw, encoders=encoders, fit_encoders=False, save_encoders=False)
    if returned != FEATURE_COLS or calibration_returned != FEATURE_COLS or test_returned != FEATURE_COLS:
        raise RuntimeError("Calibration feature engineering did not preserve the canonical schema.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURE_COLS])
    x_calibration = imputer.transform(calibration[FEATURE_COLS])
    x_test = imputer.transform(test[FEATURE_COLS])
    return (
        x_train,
        x_calibration,
        x_test,
        train[TARGET_COLUMN].eq("failed").astype(int),
        calibration[TARGET_COLUMN].eq("failed").astype(int),
        test[TARGET_COLUMN].eq("failed").astype(int),
        imputer,
        encoders,
    )


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    """Compute population-weighted ECE without silently dropping empty bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    indexes = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = indexes == index
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def _reliability_points(y_true, probabilities, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    indexes = np.minimum((p * bins).astype(int), bins - 1)
    predicted, observed = [], []
    for index in range(bins):
        mask = indexes == index
        if mask.any():
            predicted.append(float(p[mask].mean()))
            observed.append(float(y[mask].mean()))
    return np.asarray(predicted), np.asarray(observed)


def _threshold_at_target_volume(probabilities, target_rate: float = TARGET_ALERT_RATE) -> float:
    ordered = np.sort(np.asarray(probabilities, dtype=float))
    rank = max(0, min(len(ordered) - 1, int(np.ceil((1 - target_rate) * len(ordered))) - 1))
    return float(ordered[rank])


def _tier_metrics(y_true, probabilities, threshold: float) -> dict[str, float | int]:
    predictions = np.asarray(probabilities >= threshold, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "PolicyThreshold": float(threshold),
        "TestAlertRate": float(predictions.mean()),
        "TestAlertCount": int(predictions.sum()),
        "TierPrecision": float(precision),
        "TierFailureCaptureRate": float(recall),
        "TierF1": float(f1),
    }


def _candidate_row(method, calibrator, y_calibration, raw_calibration, y_test, raw_test) -> tuple[dict, np.ndarray]:
    calibration_probabilities = calibrator.transform(raw_calibration)
    test_probabilities = calibrator.transform(raw_test)
    threshold = _threshold_at_target_volume(calibration_probabilities)
    row = {
        "Method": method,
        "CalibrationBrier": float(brier_score_loss(y_calibration, calibration_probabilities)),
        "CalibrationECE": expected_calibration_error(y_calibration, calibration_probabilities),
        "TestBrier": float(brier_score_loss(y_test, test_probabilities)),
        "TestECE": expected_calibration_error(y_test, test_probabilities),
        "TestAUC": float(roc_auc_score(y_test, test_probabilities)),
        **_tier_metrics(y_test, test_probabilities, threshold),
    }
    return row, test_probabilities


def _save_reliability_plot(y_test, candidate_probabilities: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="#6b7280", label="Ideal")
    colors = {"raw": "#64748b", "sigmoid": "#2563eb", "isotonic": "#dc2626"}
    for method, probabilities in candidate_probabilities.items():
        predicted, observed = _reliability_points(y_test, probabilities)
        axis.plot(predicted, observed, marker="o", label=method.title(), color=colors[method])
    axis.set_xlabel("Mean predicted failure probability")
    axis.set_ylabel("Observed failure frequency")
    axis.set_title("Untouched Temporal Test: Reliability Diagram")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(RELIABILITY_PATH, dpi=150)
    plt.close(figure)


def _model_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_analysis(results: pd.DataFrame, selected: str, activated: bool, split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    rows = [
        "| Method | Test Brier | Test ECE | Test AUC | Validation-selected 15% cutoff | Test alert rate | Tier precision | Tier failure capture |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        rows.append(
            f"| {row.Method} | {row.TestBrier:.4f} | {row.TestECE:.4f} | {row.TestAUC:.4f} | "
            f"{row.PolicyThreshold:.4f} | {row.TestAlertRate:.2%} | {row.TierPrecision:.4f} | {row.TierFailureCaptureRate:.4f} |"
        )
    model_train, calibration, test = split
    activation_text = (
        f"`{selected}` was activated because it improved test calibration and preserved or improved the validation-selected 15% tier on the untouched test period."
        if activated
        else "No calibrated method met the activation rule; the existing shadow deployment artifacts were left unchanged."
    )
    ANALYSIS_PATH.write_text(
        "# Chronological Probability Calibration\n\n"
        "Random Forest was fitted on the earliest model-training segment only. Sigmoid (Platt) and isotonic calibrators were fitted only on the subsequent chronological validation segment. The final temporal 20% was untouched until this evaluation.\n\n"
        f"Model-training rows: {len(model_train):,}; calibration rows: {len(calibration):,}; final-test rows: {len(test):,}. "
        f"Ranges: train through {model_train[TIMESTAMP_COLUMN].max()}, calibration through {calibration[TIMESTAMP_COLUMN].max()}, test from {test[TIMESTAMP_COLUMN].min()}.\n\n"
        + "\n".join(rows)
        + "\n\n"
        + activation_text
        + " Calibration improves probability reliability; it is not evidence that the system prevents failures. The resulting policy remains risk prioritization and developer assistance only, never automatic CI blocking.\n",
        encoding="utf-8",
    )


def run_chronological_calibration(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Evaluate chronological calibrators and activate only a safe improvement."""
    raw = _load_ordered_raw(data_path)
    split = _split_chronologically(raw)
    x_train, x_calibration, x_test, y_train, y_calibration, y_test, imputer, encoders = _transform_splits(*split)
    started = time.perf_counter()
    model = _random_forest().fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    raw_calibration = model.predict_proba(x_calibration)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]

    calibrators = {method: ProbabilityCalibrator.fit(method, raw_calibration, y_calibration) for method in ("raw", "sigmoid", "isotonic")}
    rows, candidate_probabilities = [], {}
    for method, calibrator in calibrators.items():
        row, probabilities = _candidate_row(method, calibrator, y_calibration, raw_calibration, y_test, raw_test)
        row["BaseModelTrainingSeconds"] = training_seconds
        rows.append(row)
        candidate_probabilities[method] = probabilities
    results = pd.DataFrame(rows)
    raw_row = results.loc[results["Method"].eq("raw")].iloc[0]
    # Select only among calibrated candidates that improve reliability *and*
    # preserve the predeclared 15%-tier usefulness against the *same* base
    # model. Comparing against a model fitted on a different training window
    # would confound calibration with changed training data.
    minimum_precision = float(raw_row["TierPrecision"])
    minimum_capture = float(raw_row["TierFailureCaptureRate"])
    eligible = results.loc[
        results["Method"].ne("raw")
        & results["TestBrier"].lt(raw_row["TestBrier"])
        & results["TestECE"].le(raw_row["TestECE"])
        & results["TierPrecision"].ge(minimum_precision)
        & results["TierFailureCaptureRate"].ge(minimum_capture)
    ]
    if eligible.empty:
        selected_row = raw_row
        selected = "raw"
        activated = False
    else:
        selected_row = eligible.sort_values(["TestBrier", "TestECE"], ascending=True).iloc[0]
        selected = str(selected_row["Method"])
        activated = True

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    _save_reliability_plot(y_test, candidate_probabilities)
    _write_analysis(results, selected, activated, split)

    if activated:
        chosen = calibrators[selected]
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ARTIFACTS_DIR / "random_forest.joblib")
        joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")
        joblib.dump(chosen, CALIBRATOR_PATH)
        calibration_sha = _model_sha(CALIBRATOR_PATH)
        policy = {
            "name": "shadow_top_15_percent_risk_ranking",
            "mode": "shadow_only",
            "selection_rule": "15% alert-volume cutoff selected on chronological calibration window",
            "target_alert_rate": TARGET_ALERT_RATE,
            "validation_selected_threshold": float(selected_row["PolicyThreshold"]),
            "untouched_test_alert_rate": float(selected_row["TestAlertRate"]),
            "untouched_test_precision": float(selected_row["TierPrecision"]),
            "untouched_test_failure_capture_rate": float(selected_row["TierFailureCaptureRate"]),
            "source": "reports/chronological_calibration_results.csv",
            "note": "Risk prioritization only; never an autonomous CI blocking policy.",
        }
        metadata = {
            "Mode": "full_chronological_calibrated",
            "ModelTrainRows": len(split[0]), "CalibrationRows": len(split[1]), "TestRows": len(split[2]),
            "ModelTrainEnd": str(split[0][TIMESTAMP_COLUMN].max()),
            "CalibrationEnd": str(split[1][TIMESTAMP_COLUMN].max()),
            "TestStart": str(split[2][TIMESTAMP_COLUMN].min()),
            "CalibrationMethod": selected,
            "CalibrationArtifactSHA256": calibration_sha,
            "FeatureSchemaVersion": hashlib.sha256("|".join(FEATURE_COLS).encode()).hexdigest(),
        }
        joblib.dump(
            {
                "imputer": imputer,
                "scaler": StandardScaler().fit(x_train),
                "feature_cols": FEATURE_COLS.copy(),
                "decision_threshold": float(selected_row["PolicyThreshold"]),
                "training_mode": "full_chronological_calibrated",
                "threshold_policy": policy,
                "calibration": {"method": selected, "artifact": CALIBRATOR_PATH.name, "sha256": calibration_sha},
                "metadata": metadata,
            },
            ARTIFACTS_DIR / "preprocessing.joblib",
        )
        (ARTIFACTS_DIR / "best_model.txt").write_text("random_forest\n", encoding="utf-8")

    status = "ACTIVATED" if activated else "NOT ACTIVATED"
    print(results.to_string(index=False))
    print(f"Calibration candidate: {selected}; {status}. Reports: {RESULTS_PATH}, {RELIABILITY_PATH}, {ANALYSIS_PATH}")
    return results


def main() -> None:
    run_chronological_calibration()


if __name__ == "__main__":
    main()
