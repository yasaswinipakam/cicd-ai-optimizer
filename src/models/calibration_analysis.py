"""Research-only probability calibration analysis for the frozen Random Forest.

The final 20% production test split is never passed to either calibration method.
Platt and isotonic calibration use five-fold out-of-fold predictions within the
original training partition, so their probability mapping is learned without
double-dipping on the reporting set.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import joblib

# Keep matplotlib usable in non-interactive and sandboxed environments.
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "cicd-ai-optimizer-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.schema import FEATURE_COLS
from src.models.evaluate import _recreate_split
from src.models.train import ARTIFACTS_DIR, DATA_PATH, RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "calibration_results.csv"
FIGURE_PATH = REPORTS_DIR / "reliability_diagram.png"
ANALYSIS_PATH = REPORTS_DIR / "calibration_analysis.md"
N_BINS = 10


def _random_forest() -> RandomForestClassifier:
    """Return the frozen production Random Forest configuration."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def expected_calibration_error(y_true, probabilities, n_bins: int = N_BINS) -> float:
    """Compute equal-width-bin ECE, weighting each bin by its sample share."""
    truth = pd.Series(y_true).to_numpy(dtype=float)
    predicted = pd.Series(probabilities).to_numpy(dtype=float)
    bin_index = pd.cut(predicted, bins=n_bins, labels=False, include_lowest=True)
    ece = 0.0
    for bin_number in range(n_bins):
        selected = bin_index == bin_number
        if not selected.any():
            continue
        observed_rate = truth[selected].mean()
        mean_probability = predicted[selected].mean()
        ece += (selected.sum() / len(predicted)) * abs(observed_rate - mean_probability)
    return float(ece)


def _metrics(y_test, probabilities) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "BrierScore": float(brier_score_loss(y_test, probabilities)),
        "ECE": expected_calibration_error(y_test, probabilities),
        "F1": float(f1_score(y_test, predictions, zero_division=0)),
        "AUC": float(roc_auc_score(y_test, probabilities)),
    }


def _calibrated_estimator(method: str) -> CalibratedClassifierCV:
    """Create a model whose calibration folds remain entirely within training data."""
    # Imputation is inside each fold rather than being fitted across the final test
    # set or its labels.  Every clone keeps the production RF hyperparameters.
    base_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("random_forest", _random_forest()),
        ]
    )
    calibration_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv=calibration_cv,
        n_jobs=1,
    )


def _save_reliability_diagram(y_test, probability_sets: dict[str, object]) -> None:
    """Save raw, Platt, and isotonic reliability curves with the ideal diagonal."""
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Ideal calibration")
    colors = {"Raw RF": "#4c78a8", "Platt": "#f58518", "Isotonic": "#54a24b"}
    for method, probabilities in probability_sets.items():
        observed, mean_predicted = calibration_curve(y_test, probabilities, n_bins=N_BINS, strategy="uniform")
        axis.plot(
            mean_predicted,
            observed,
            marker="o",
            linewidth=2,
            label=method,
            color=colors[method],
        )
    axis.set_title("Random Forest Reliability Diagram")
    axis.set_xlabel("Mean predicted failure probability")
    axis.set_ylabel("Observed failure frequency")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def _write_analysis(results: pd.DataFrame) -> None:
    best_reliability = results.loc[results["BrierScore"].idxmin()]
    raw = results.loc[results["Method"] == "Raw RF"].iloc[0]
    table = [
        "| Method | Brier Score | ECE | F1 | AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        table.append(f"| {row.Method} | {row.BrierScore:.4f} | {row.ECE:.4f} | {row.F1:.4f} | {row.AUC:.4f} |")
    ANALYSIS_PATH.write_text(
        "# Probability Calibration Analysis\n\n"
        "The final 40,000-row production test partition was not used to fit either calibration method. "
        "Platt (sigmoid) and isotonic calibration were fitted through five-fold stratified calibration within the original training partition; "
        "each fold fits imputation and the fixed Random Forest configuration only on its fitting fold.\n\n"
        + "\n".join(table)
        + "\n\n"
        + f"By Brier score, **{best_reliability.Method}** is the most reliable method in this experiment "
        + f"({best_reliability.BrierScore:.4f}; ECE={best_reliability.ECE:.4f}). "
        + f"Compared with raw RF, its F1 changes by {best_reliability.F1 - raw.F1:+.4f} and AUC by {best_reliability.AUC - raw.AUC:+.4f}. "
        + "Calibration changes the probability mapping rather than the underlying feature ranking, so little change in AUC is expected. "
        + "This is a research recommendation only; no calibrated estimator has been deployed to the API.\n",
        encoding="utf-8",
    )


def run_calibration_analysis(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Evaluate frozen raw probabilities and two train-only calibration approaches."""
    assert_features_available_for_prediction(FEATURE_COLS)
    preprocessing = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    if preprocessing["feature_cols"] != FEATURE_COLS:
        raise ValueError("Frozen preprocessing schema does not match FEATURE_COLS.")

    x_train, x_test, y_train, y_test = _recreate_split(data_path, FEATURE_COLS, encoders)
    frozen_raw_rf = joblib.load(ARTIFACTS_DIR / "random_forest.joblib")
    raw_test_input = preprocessing["imputer"].transform(x_test)
    raw_probabilities = frozen_raw_rf.predict_proba(raw_test_input)[:, 1]

    probability_sets: dict[str, object] = {"Raw RF": raw_probabilities}
    rows = [{"Method": "Raw RF", **_metrics(y_test, raw_probabilities)}]
    for label, method in (("Platt", "sigmoid"), ("Isotonic", "isotonic")):
        calibrated = _calibrated_estimator(method)
        calibrated.fit(x_train, y_train)
        probabilities = calibrated.predict_proba(x_test)[:, 1]
        probability_sets[label] = probabilities
        rows.append({"Method": label, **_metrics(y_test, probabilities)})

    results = pd.DataFrame(rows, columns=["Method", "BrierScore", "ECE", "F1", "AUC"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    _save_reliability_diagram(y_test, probability_sets)
    _write_analysis(results)
    print(results.to_string(index=False))
    print(f"Saved research-only outputs to {RESULTS_PATH}, {FIGURE_PATH}, and {ANALYSIS_PATH}")
    return results


def main() -> None:
    run_calibration_analysis()


if __name__ == "__main__":
    main()
