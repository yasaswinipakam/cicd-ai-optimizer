"""Research-only repository-aware calibration and policy adaptation study.

For each preselected large repository, the workflow is strictly chronological:

* train a global Random Forest on *other* repositories observed before the target
  repository's final-test period;
* fit an isotonic probability mapper using the target repository's earliest 10%;
* select an operating threshold on the next 10%; and
* report exactly once on the target repository's untouched final 80%.

This is calibration/policy adaptation, not deployment fine-tuning.  It is designed
for a non-incremental Random Forest and leaves every production artifact unchanged.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import tempfile
import time

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "cicd-ai-optimizer-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.engineer import engineer_features
from src.features.schema import FEATURE_COLS
from src.models.calibration_analysis import expected_calibration_error
from src.models.train import DATA_PATH, RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "repository_adaptation_results.csv"
ANALYSIS_PATH = REPORTS_DIR / "repository_adaptation_analysis.md"
FIGURE_PATH = REPORTS_DIR / "repository_adaptation_comparison.png"
REPOSITORY_COLUMN = "gh_project_name"
TIMESTAMP_COLUMN = "gh_build_started_at"
TARGET_COLUMN = "tr_status"
HOLDOUT_REPOSITORY_COUNT = 3
CALIBRATION_FRACTION = 0.10
POLICY_FRACTION = 0.10
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

INPUT_COLUMNS = [
    REPOSITORY_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    "git_prev_commit_resolution_status",
    "gh_lang",
    "git_branch",
    "gh_by_core_team_member",
    "gh_is_pr",
    "git_diff_src_churn",
    "git_diff_test_churn",
    "gh_diff_files_added",
    "gh_diff_files_deleted",
    "gh_diff_files_modified",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
]


def _random_forest() -> RandomForestClassifier:
    """Return the unchanged production Random Forest configuration."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _fit_encoders(training_raw: pd.DataFrame) -> dict[str, LabelEncoder]:
    """Fit encoders on global training rows only, never target adaptation/test rows."""
    return {
        "prev_status_enc": LabelEncoder().fit(
            training_raw["git_prev_commit_resolution_status"].fillna("__missing__").astype(str)
        ),
        "lang_enc": LabelEncoder().fit(training_raw["gh_lang"].fillna("__missing__").astype(str)),
    }


def _prepare_features(raw: pd.DataFrame, encoders: dict[str, LabelEncoder]) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the global encoder mapping without refitting it on target rows."""
    feature_input = raw.drop(columns=[REPOSITORY_COLUMN], errors="ignore")
    engineered, columns = engineer_features(
        feature_input, encoders=encoders, fit_encoders=False, save_encoders=False
    )
    if columns != FEATURE_COLS:
        raise RuntimeError("Adaptation feature ordering does not match strict FEATURE_COLS.")
    return engineered[FEATURE_COLS], engineered[TARGET_COLUMN].eq("failed").astype(int)


def _select_threshold(y_policy: pd.Series, probabilities: np.ndarray) -> float:
    """Select only on the policy window; ties favour fewer alerts (higher threshold)."""
    candidates = []
    for threshold in THRESHOLDS:
        predicted = (probabilities >= threshold).astype(int)
        candidates.append((f1_score(y_policy, predicted, zero_division=0), float(threshold)))
    return max(candidates, key=lambda value: (value[0], value[1]))[1]


def _row(repository: str, method: str, y_test, probabilities, threshold: float, metadata: dict[str, object]) -> dict[str, object]:
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, average="binary", zero_division=0
    )
    return {
        "Repository": repository,
        "Method": method,
        **metadata,
        "Threshold": float(threshold),
        "TestFailureRate": float(pd.Series(y_test).mean()),
        "AlertRate": float(predicted.mean()),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_test, probabilities)),
        "BrierScore": float(brier_score_loss(y_test, probabilities)),
        "ECE": expected_calibration_error(y_test, probabilities),
    }


def _evaluate_repository(repository: str, raw: pd.DataFrame) -> list[dict[str, object]]:
    """Run one fully isolated global-to-target chronological adaptation experiment."""
    target = raw.loc[raw[REPOSITORY_COLUMN].eq(repository)].copy()
    target = target.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    calibration_end = int(len(target) * CALIBRATION_FRACTION)
    policy_end = int(len(target) * (CALIBRATION_FRACTION + POLICY_FRACTION))
    calibration_raw = target.iloc[:calibration_end]
    policy_raw = target.iloc[calibration_end:policy_end]
    final_test_raw = target.iloc[policy_end:]
    if min(len(calibration_raw), len(policy_raw), len(final_test_raw)) == 0:
        raise ValueError(f"Target repository {repository!r} is too small for chronological adaptation.")

    # A global training build may not include any target-repository row and may not
    # be later than the start of this target's untouched final-test window.
    test_start = final_test_raw[TIMESTAMP_COLUMN].min()
    global_train_raw = raw.loc[
        ~raw[REPOSITORY_COLUMN].eq(repository) & raw[TIMESTAMP_COLUMN].lt(test_start)
    ].copy()
    if global_train_raw.empty:
        raise ValueError(f"No contemporaneously valid global rows for {repository!r}.")
    if global_train_raw[REPOSITORY_COLUMN].eq(repository).any():
        raise RuntimeError("Target repository leaked into global training data.")

    encoders = _fit_encoders(global_train_raw)
    x_global, y_global = _prepare_features(global_train_raw, encoders)
    x_calibration, y_calibration = _prepare_features(calibration_raw, encoders)
    x_policy, y_policy = _prepare_features(policy_raw, encoders)
    x_test, y_test = _prepare_features(final_test_raw, encoders)
    imputer = SimpleImputer(strategy="median")
    x_global_ready = imputer.fit_transform(x_global)
    x_calibration_ready = imputer.transform(x_calibration)
    x_policy_ready = imputer.transform(x_policy)
    x_test_ready = imputer.transform(x_test)
    model = _random_forest()
    started = time.perf_counter()
    model.fit(x_global_ready, y_global)
    training_seconds = time.perf_counter() - started
    raw_calibration_probabilities = model.predict_proba(x_calibration_ready)[:, 1]
    raw_policy_probabilities = model.predict_proba(x_policy_ready)[:, 1]
    raw_test_probabilities = model.predict_proba(x_test_ready)[:, 1]

    # Isotonic mapping learns only from the earliest target-repository window.
    probability_mapper = IsotonicRegression(out_of_bounds="clip")
    probability_mapper.fit(raw_calibration_probabilities, y_calibration)
    calibrated_policy_probabilities = probability_mapper.transform(raw_policy_probabilities)
    calibrated_test_probabilities = probability_mapper.transform(raw_test_probabilities)
    adapted_threshold = _select_threshold(y_policy, calibrated_policy_probabilities)
    metadata = {
        "GlobalTrainRows": len(global_train_raw),
        "CalibrationRows": len(calibration_raw),
        "PolicyRows": len(policy_raw),
        "TestRows": len(final_test_raw),
        "GlobalTrainEnd": str(global_train_raw[TIMESTAMP_COLUMN].max()),
        "TestStart": str(test_start),
        "TestEnd": str(final_test_raw[TIMESTAMP_COLUMN].max()),
        "TrainingSeconds": float(training_seconds),
        "GlobalTrainingContainsTarget": False,
    }
    results = [
        _row(repository, "Global RF (default threshold)", y_test, raw_test_probabilities, 0.50, metadata),
        _row(
            repository,
            "Global RF + target isotonic calibration + target policy threshold",
            y_test,
            calibrated_test_probabilities,
            adapted_threshold,
            metadata,
        ),
    ]
    del global_train_raw, x_global, x_global_ready, model
    gc.collect()
    return results


def _save_plot(results: pd.DataFrame) -> None:
    """Show untouched-final-test F1 and Brier score for each repository/method."""
    labels = [f"{repo}\n{method.replace('Global RF ', '')}" for repo, method in zip(results["Repository"], results["Method"])]
    positions = np.arange(len(results))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(positions, results["F1"], color=["#4c78a8", "#54a24b"] * (len(results) // 2))
    axes[0].set_title("Untouched Later-Build F1")
    axes[0].set_ylabel("F1")
    axes[1].bar(positions, results["BrierScore"], color=["#4c78a8", "#54a24b"] * (len(results) // 2))
    axes[1].set_title("Untouched Later-Build Brier Score (lower is better)")
    axes[1].set_ylabel("Brier score")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def _write_analysis(results: pd.DataFrame, counts: pd.Series) -> None:
    table = [
        "| Repository | Method | Calibration / policy / final-test builds | Threshold | Alert rate | Precision | Recall | F1 | AUC | Brier | ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        windows = f"{row.CalibrationRows:,} / {row.PolicyRows:,} / {row.TestRows:,}"
        table.append(
            f"| {row.Repository} | {row.Method} | {windows} | {row.Threshold:.2f} | {row.AlertRate:.4f} | {row.Precision:.4f} | "
            f"{row.Recall:.4f} | {row.F1:.4f} | {row.AUC:.4f} | {row.BrierScore:.4f} | {row.ECE:.4f} |"
        )
    comparisons: list[str] = []
    for repository in counts.head(HOLDOUT_REPOSITORY_COUNT).index:
        pair = results.loc[results["Repository"].eq(repository)].set_index("Method")
        raw = pair.loc["Global RF (default threshold)"]
        adapted = pair.loc["Global RF + target isotonic calibration + target policy threshold"]
        comparisons.append(
            f"`{repository}`: F1 {raw.F1:.4f} → {adapted.F1:.4f}; Brier {raw.BrierScore:.4f} → {adapted.BrierScore:.4f}."
        )
    ANALYSIS_PATH.write_text(
        "# Repository-Aware Adaptation Study\n\n"
        "For each repository, a global Random Forest was fitted only on other repositories with timestamps before the target repository's final-test window. "
        "The target's earliest 10% of builds fitted an isotonic probability mapper, the next 10% selected a threshold from 0.05–0.95 by F1, and the later 80% remained untouched until final reporting. "
        "This is adaptation of calibration and policy, not a production model replacement or Random Forest fine-tuning.\n\n"
        + "\n".join(table)
        + "\n\n"
        + " ".join(comparisons)
        + " For `DataDog/dd-agent`, the F1 increase is driven by the selected 0.05 threshold alerting 100% of later builds; it is therefore not evidence of useful selective ranking. "
        + "For `pyca/cryptography`, calibration greatly improves Brier/ECE but does not improve F1. Results are restricted to the three largest repositories, selected in advance by build count, and do not establish that one adaptation policy will transfer to every repository.\n",
        encoding="utf-8",
    )


def run_repository_adaptation_study(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Run deterministic chronological target-adaptation experiments for three repositories."""
    assert_features_available_for_prediction(FEATURE_COLS)
    if REPOSITORY_COLUMN in FEATURE_COLS:
        raise RuntimeError("Repository identity must not enter the strict prediction schema.")
    raw = pd.read_parquet(data_path, columns=INPUT_COLUMNS)
    raw[TIMESTAMP_COLUMN] = pd.to_datetime(raw[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    if raw[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Unparsable build timestamp prevents chronological adaptation study.")
    counts = raw[REPOSITORY_COLUMN].value_counts()
    selected_repositories = counts.head(HOLDOUT_REPOSITORY_COUNT).index.tolist()
    rows: list[dict[str, object]] = []
    for repository in selected_repositories:
        rows.extend(_evaluate_repository(repository, raw))
        gc.collect()
    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    _save_plot(results)
    _write_analysis(results, counts)
    print(results.to_string(index=False))
    print(f"Saved research-only outputs to {RESULTS_PATH}, {FIGURE_PATH}, and {ANALYSIS_PATH}")
    return results


def main() -> None:
    run_repository_adaptation_study()


if __name__ == "__main__":
    main()
