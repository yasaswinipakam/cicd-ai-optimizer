"""Activate a measured, non-blocking risk-ranking policy for shadow deployment.

This utility intentionally updates *only* the decision-policy metadata in the
preprocessing bundle.  It never retrains or replaces the Random Forest,
encoders, imputer, or feature schema.  The chosen cutoff comes from the frozen
full-temporal alert-volume study and is suitable for collecting live outcome
data, not for automatically blocking CI builds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "preprocessing.joblib"
POLICY_RESULTS_PATH = PROJECT_ROOT / "reports" / "full_temporal_alert_policy_results.csv"


def activate_shadow_policy(target_alert_rate: float = 0.15) -> dict:
    """Apply a measured top-risk cutoff from the fixed-volume policy report."""
    results = pd.read_csv(POLICY_RESULTS_PATH)
    matching = results.loc[(results["TargetAlertRate"] - target_alert_rate).abs() < 1e-9]
    if matching.empty:
        available = ", ".join(f"{value:.0%}" for value in results["TargetAlertRate"])
        raise ValueError(f"No measured policy for {target_alert_rate:.0%}; available: {available}.")

    row = matching.iloc[0]
    bundle = joblib.load(ARTIFACT_PATH)
    if bundle.get("training_mode") != "full":
        raise ValueError("Shadow policy activation requires full-mode temporal model artifacts.")

    prior_threshold = float(bundle.get("decision_threshold", 0.5))
    bundle["decision_threshold"] = float(row["ScoreThreshold"])
    bundle["threshold_policy"] = {
        "name": f"shadow_top_{int(target_alert_rate * 100)}_percent_risk_ranking",
        "mode": "shadow_only",
        "selection_rule": "fixed top-risk alert volume on untouched full temporal test period",
        "target_alert_rate": float(row["TargetAlertRate"]),
        "measured_alert_rate": float(row["AlertRate"]),
        "measured_precision": float(row["Precision"]),
        "measured_failure_capture_rate": float(row["FailureCaptureRate"]),
        "source": "reports/full_temporal_alert_policy_results.csv",
        "note": "Risk-ranking developer assistance only; never an autonomous CI blocking policy.",
        "previous_validation_threshold": prior_threshold,
    }
    metadata = dict(bundle.get("metadata", {}))
    metadata["ActiveShadowDecisionThreshold"] = float(row["ScoreThreshold"])
    metadata["ActiveShadowTargetAlertRate"] = float(row["TargetAlertRate"])
    metadata["TrainingValidationThreshold"] = prior_threshold
    bundle["metadata"] = metadata
    joblib.dump(bundle, ARTIFACT_PATH)
    return bundle["threshold_policy"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate a measured shadow alert-volume policy.")
    parser.add_argument("--target-alert-rate", type=float, default=0.15)
    args = parser.parse_args()
    policy = activate_shadow_policy(args.target_alert_rate)
    print(
        f"Activated {policy['name']} at score >= {float(joblib.load(ARTIFACT_PATH)['decision_threshold']):.6f}; "
        f"measured temporal alert rate={policy['measured_alert_rate']:.2%}."
    )


if __name__ == "__main__":
    main()
