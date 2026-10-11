"""Operating policies at fixed alert volumes for the deployed full temporal RF.

This is a read-only analysis: it scores the frozen deployable model against its
existing untouched chronological test period. It neither refits nor overwrites any
artifact, and it does not change the live threshold.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import joblib

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "cicd-ai-optimizer-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.features.engineer import engineer_features
from src.features.schema import FEATURE_COLS, FEATURE_SOURCE_COLUMNS
from src.models.train import ARTIFACTS_DIR, DATA_PATH, FULL_TRAIN_FRACTION, TARGET_COLUMN, TIMESTAMP_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "full_temporal_alert_policy_results.csv"
ANALYSIS_PATH = REPORTS_DIR / "full_temporal_alert_policy.md"
FIGURE_PATH = REPORTS_DIR / "full_temporal_alert_policy.png"
TARGET_ALERT_RATES = (0.05, 0.10, 0.15, 0.20, 0.35)


def _test_probabilities(data_path: Path = DATA_PATH) -> tuple[pd.Series, np.ndarray]:
    """Recreate only the final chronological test period for frozen artifact scoring."""
    bundle = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
    if bundle.get("training_mode") != "full":
        raise ValueError("This analysis requires artifacts produced by `train --mode full`.")
    if bundle["feature_cols"] != FEATURE_COLS:
        raise ValueError("Frozen artifact schema does not match the current strict schema.")
    columns = list(dict.fromkeys([TARGET_COLUMN, *FEATURE_SOURCE_COLUMNS]))
    raw = pd.read_parquet(data_path, columns=columns, engine="pyarrow")
    raw[TIMESTAMP_COLUMN] = pd.to_datetime(raw[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    if raw[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Chronological policy analysis requires valid build-start timestamps.")
    ordered = raw.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
    test_raw = ordered.iloc[int(len(ordered) * FULL_TRAIN_FRACTION):]
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    model = joblib.load(ARTIFACTS_DIR / "random_forest.joblib")
    engineered, returned_columns = engineer_features(
        test_raw, encoders=encoders, fit_encoders=False, save_encoders=False
    )
    if returned_columns != FEATURE_COLS:
        raise RuntimeError("Feature engineering returned an unexpected schema.")
    y_test = engineered[TARGET_COLUMN].eq("failed").astype(int)
    probabilities = model.predict_proba(bundle["imputer"].transform(engineered[FEATURE_COLS]))[:, 1]
    return y_test, probabilities


def _threshold_for_target_rate(probabilities: np.ndarray, target_rate: float) -> float:
    """Choose a score cutoff near the requested top-risk volume.

    Score ties mean a single threshold cannot guarantee an exact alert count; the
    achieved rate is always reported rather than silently claiming the target rate.
    """
    rank = max(0, min(len(probabilities) - 1, int(np.ceil((1 - target_rate) * len(probabilities))) - 1))
    return float(np.sort(probabilities)[rank])


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    alerts = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, alerts, average="binary", zero_division=0
    )
    return {
        "ScoreThreshold": float(threshold),
        "AlertCount": int(alerts.sum()),
        "AlertRate": float(alerts.mean()),
        "Precision": float(precision),
        "FailureCaptureRate": float(recall),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_true, probabilities)),
    }


def _save_plot(results: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    labels = [f"{rate:.0%}" for rate in results["TargetAlertRate"]]
    positions = np.arange(len(results))
    width = 0.27
    axis.bar(positions - width, results["Precision"], width, label="Precision", color="#4c78a8")
    axis.bar(positions, results["FailureCaptureRate"], width, label="Failure capture", color="#e45756")
    axis.bar(positions + width, results["AlertRate"], width, label="Achieved alert rate", color="#54a24b")
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Target top-risk alert volume")
    axis.set_ylabel("Rate")
    axis.set_ylim(0, 1)
    axis.set_title("Full Temporal Model: Alert-Volume Trade-offs")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def _write_analysis(results: pd.DataFrame) -> None:
    table = [
        "| Target alert rate | Score threshold | Achieved alert rate | Precision | Failure capture | F1 | Alerts |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        table.append(
            f"| {row.TargetAlertRate:.0%} | {row.ScoreThreshold:.4f} | {row.AlertRate:.2%} | "
            f"{row.Precision:.4f} | {row.FailureCaptureRate:.4f} | {row.F1:.4f} | {row.AlertCount:,} |"
        )
    policy = results.loc[results["TargetAlertRate"].eq(0.15)].iloc[0]
    ANALYSIS_PATH.write_text(
        "# Full Temporal Alert-Volume Policy Analysis\n\n"
        "This analysis uses only frozen full-mode Random Forest probabilities for the untouched later 20% chronological test period. "
        "It does not retrain, calibrate, or change the live threshold. Because raw probabilities have not yet been recalibrated "
        "for this full temporal model, score cutoffs should be treated as risk-ranking policies rather than literal failure likelihoods.\n\n"
        + "\n".join(table)
        + "\n\n"
        + f"For an initial shadow deployment, the 15% top-risk policy (score ≥ {policy.ScoreThreshold:.4f}) is a practical review starting point: "
        + f"it produces alerts for {policy.AlertRate:.1%} of later builds and captures {policy.FailureCaptureRate:.1%} of failures. "
        + "It is a recommendation for measuring operational utility, not an automatic build-blocking rule.\n",
        encoding="utf-8",
    )


def run_full_temporal_alert_policy_analysis() -> pd.DataFrame:
    """Report five predeclared top-risk operating policies without modifying artifacts."""
    y_test, probabilities = _test_probabilities()
    rows: list[dict[str, float | int]] = []
    for rate in TARGET_ALERT_RATES:
        threshold = _threshold_for_target_rate(probabilities, rate)
        rows.append({"TargetAlertRate": rate, **_metrics(y_test, probabilities, threshold)})
    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    _save_plot(results)
    _write_analysis(results)
    print(results.to_string(index=False))
    print(f"Saved policy analysis to {RESULTS_PATH}, {FIGURE_PATH}, and {ANALYSIS_PATH}")
    return results


def main() -> None:
    run_full_temporal_alert_policy_analysis()


if __name__ == "__main__":
    main()
