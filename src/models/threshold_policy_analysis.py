"""Research-only operating-threshold analysis of frozen RF test probabilities.

This module deliberately uses the persisted production Random Forest and its
existing deterministic held-out test partition.  No estimator, calibrator, or API
configuration is fitted or modified.  Although R6 found isotonic calibration more
reliable, that research-only calibrator was not persisted; recreating it here would
require new model fitting and violate this analysis' no-retraining scope.
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
from sklearn.metrics import precision_recall_fscore_support

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.schema import FEATURE_COLS
from src.models.evaluate import _recreate_split
from src.models.train import ARTIFACTS_DIR, DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "threshold_sweep_results.csv"
FIGURE_PATH = REPORTS_DIR / "threshold_tradeoff_plot.png"
ANALYSIS_PATH = REPORTS_DIR / "decision_policy_analysis.md"
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)


def _threshold_metrics(y_true, probabilities, threshold: float) -> dict[str, float | int]:
    """Calculate the operational outcome of flagging probability >= threshold."""
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "Threshold": float(threshold),
        "AlertCount": int(predictions.sum()),
        "AlertRate": float(predictions.mean()),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        # This duplicate is intentional: it communicates recall in operational terms.
        "FailureCaptureRate": float(recall),
    }


def _select_policies(sweep: pd.DataFrame) -> pd.DataFrame:
    """Use predeclared rules to choose three non-objective operating policies."""
    # Ties favour the higher threshold, which reduces alerts without lowering F1.
    balanced = sweep.loc[sweep["F1"].eq(sweep["F1"].max())].sort_values("Threshold", ascending=False).iloc[0]

    # Select the highest threshold that captures at least 85% of failures. This
    # expresses the product choice without optimising on a hidden business cost.
    catch_candidates = sweep.loc[sweep["Recall"] >= 0.85]
    catch_failures = (
        catch_candidates.sort_values("Threshold", ascending=False).iloc[0]
        if not catch_candidates.empty
        else sweep.sort_values(["Recall", "Threshold"], ascending=[False, False]).iloc[0]
    )

    # At a higher threshold than Balanced, choose the point with greatest precision;
    # ties favour fewer alerts.  A 1% minimum alert rate excludes unstable
    # near-zero-alert points (for example, four alerts with 100% precision).
    alarm_candidates = sweep.loc[
        (sweep["Threshold"] > balanced["Threshold"]) & (sweep["AlertRate"] >= 0.01)
    ]
    avoid_false_alarms = (
        alarm_candidates.sort_values(["Precision", "Threshold"], ascending=[False, False]).iloc[0]
        if not alarm_candidates.empty
        else balanced
    )
    selected = pd.DataFrame(
        [
            {"Policy": "Balanced", **balanced.to_dict()},
            {"Policy": "Catch failures", **catch_failures.to_dict()},
            {"Policy": "Avoid false alarms", **avoid_false_alarms.to_dict()},
        ]
    )
    return selected


def _save_plot(sweep: pd.DataFrame, policies: pd.DataFrame) -> None:
    """Plot precision, recall/capture, and alert rate across operating thresholds."""
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(sweep["Threshold"], sweep["Precision"], marker="o", label="Precision", color="#4c78a8")
    axis.plot(sweep["Threshold"], sweep["Recall"], marker="o", label="Recall / failure capture", color="#e45756")
    axis.plot(sweep["Threshold"], sweep["AlertRate"], marker="o", label="Alert rate", color="#54a24b")
    for row in policies.itertuples(index=False):
        axis.axvline(row.Threshold, color="#777777", linestyle=":", alpha=0.6)
        axis.annotate(row.Policy, (row.Threshold, 0.02), rotation=90, va="bottom", ha="right", fontsize=8)
    axis.set_title("Frozen Random Forest Threshold Trade-offs")
    axis.set_xlabel("Failure-risk alert threshold")
    axis.set_ylabel("Metric")
    axis.set_xlim(0.05, 0.95)
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def _write_analysis(sweep: pd.DataFrame, policies: pd.DataFrame) -> None:
    policy_table = [
        "| Policy | Threshold | Precision | Recall | Alert rate | Failure capture rate | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in policies.itertuples(index=False):
        policy_table.append(
            f"| {row.Policy} | {row.Threshold:.2f} | {row.Precision:.4f} | {row.Recall:.4f} | "
            f"{row.AlertRate:.4f} | {row.FailureCaptureRate:.4f} | {row.F1:.4f} |"
        )
    balanced = policies.loc[policies["Policy"].eq("Balanced")].iloc[0]
    catch = policies.loc[policies["Policy"].eq("Catch failures")].iloc[0]
    alarms = policies.loc[policies["Policy"].eq("Avoid false alarms")].iloc[0]
    ANALYSIS_PATH.write_text(
        "# Decision Threshold Policy Analysis\n\n"
        "This sweep uses only probabilities from the frozen production Random Forest on its existing deterministic 40,000-row test split. "
        "No estimator, calibrator, preprocessing object, artifact, or API threshold was changed. The R6 isotonic calibrator is not used here because it was research-only and not persisted; recreating it would require retraining.\n\n"
        + "\n".join(policy_table)
        + "\n\n"
        + f"For a developer warning before merge, **Balanced** (threshold {balanced.Threshold:.2f}) is the most reasonable starting recommendation because it maximizes F1 on this held-out split. "
        + f"If missed failures are more costly, **Catch failures** raises capture to {catch.FailureCaptureRate:.1%} but alerts on {catch.AlertRate:.1%} of builds. "
        + f"If alert fatigue is the dominant concern, **Avoid false alarms** raises precision to {alarms.Precision:.1%} while reducing captured failures to {alarms.FailureCaptureRate:.1%}. "
        + "These are product trade-offs, not objectively correct settings; a product owner should set the operating policy using the cost of missed failures and false alerts.\n",
        encoding="utf-8",
    )


def run_threshold_policy_analysis(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep thresholds over immutable held-out production test probabilities."""
    assert_features_available_for_prediction(FEATURE_COLS)
    preprocessing = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    model = joblib.load(ARTIFACTS_DIR / "random_forest.joblib")
    if preprocessing["feature_cols"] != FEATURE_COLS:
        raise ValueError("Frozen preprocessing schema does not match FEATURE_COLS.")
    _x_train, x_test, _y_train, y_test = _recreate_split(data_path, FEATURE_COLS, encoders)
    probabilities = model.predict_proba(preprocessing["imputer"].transform(x_test))[:, 1]
    sweep = pd.DataFrame([_threshold_metrics(y_test, probabilities, threshold) for threshold in THRESHOLDS])
    policies = _select_policies(sweep)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(RESULTS_PATH, index=False)
    _save_plot(sweep, policies)
    _write_analysis(sweep, policies)
    print("Threshold sweep:\n" + sweep.to_string(index=False))
    print("\nNamed policies:\n" + policies.to_string(index=False))
    print(f"Saved research-only outputs to {RESULTS_PATH}, {FIGURE_PATH}, and {ANALYSIS_PATH}")
    return sweep, policies


def main() -> None:
    run_threshold_policy_analysis()


if __name__ == "__main__":
    main()
