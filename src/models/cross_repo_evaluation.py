"""Research-only leave-one-repository-out evaluation for CI failure prediction.

Repository identity is deliberately excluded from ``FEATURE_COLS``.  This script
uses the three largest repositories by build count as an outcome-independent
hold-out rule, fits encoders/imputation/models only on the remaining repositories,
and writes reports without modifying production artifacts.
"""

from __future__ import annotations

import gc
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.engineer import engineer_features
from src.features.schema import FEATURE_COLS
from src.models.evaluate import _recreate_split
from src.models.train import ARTIFACTS_DIR, DATA_PATH, RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "cross_repo_results.csv"
ANALYSIS_PATH = REPORTS_DIR / "cross_repo_analysis.md"
REPOSITORY_COLUMN = "gh_project_name"
TARGET_COLUMN = "tr_status"
HOLDOUT_REPOSITORY_COUNT = 3

# Only columns necessary for strict feature engineering, target, and repository
# separation are read. Repository name is never supplied to the model matrix.
INPUT_COLUMNS = [
    REPOSITORY_COLUMN,
    TARGET_COLUMN,
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
    """Return the exact frozen production Random Forest configuration."""
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
    """Fit categorical encoders from non-held-out repositories only."""
    sources = {
        "prev_status_enc": "git_prev_commit_resolution_status",
        "lang_enc": "gh_lang",
    }
    return {
        feature: LabelEncoder().fit(training_raw[source].fillna("__missing__").astype(str))
        for feature, source in sources.items()
    }


def _evaluate_holdout(repository: str, raw: pd.DataFrame) -> dict[str, object]:
    """Fit on every other repository and evaluate the named repository only."""
    test_raw = raw.loc[raw[REPOSITORY_COLUMN].eq(repository)].drop(columns=[REPOSITORY_COLUMN]).copy()
    train_raw = raw.loc[~raw[REPOSITORY_COLUMN].eq(repository)].drop(columns=[REPOSITORY_COLUMN]).copy()
    if len(test_raw) == 0 or len(train_raw) == 0:
        raise ValueError(f"Repository split invalid for {repository!r}.")
    # Guard against accidental project identity leakage before fitting anything.
    if REPOSITORY_COLUMN in train_raw.columns or REPOSITORY_COLUMN in test_raw.columns:
        raise RuntimeError("Repository identifier reached the feature-engineering input.")

    encoders = _fit_encoders(train_raw)
    train_features, feature_columns = engineer_features(
        train_raw, encoders=encoders, fit_encoders=False, save_encoders=False
    )
    test_features, _ = engineer_features(
        test_raw, encoders=encoders, fit_encoders=False, save_encoders=False
    )
    if feature_columns != FEATURE_COLS:
        raise RuntimeError("Cross-repository feature order does not match FEATURE_COLS.")

    y_train = train_features[TARGET_COLUMN].eq("failed").astype(int)
    y_test = test_features[TARGET_COLUMN].eq("failed").astype(int)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train_features[FEATURE_COLS])
    x_test = imputer.transform(test_features[FEATURE_COLS])
    model = _random_forest()
    started = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    top_language = str(test_raw["gh_lang"].fillna("__missing__").mode().iloc[0])
    return {
        "Evaluation": f"Held-out: {repository}",
        "Repository": repository,
        "TrainRows": len(train_raw),
        "TestRows": len(test_raw),
        "TestFailureRate": float(y_test.mean()),
        "TopTestLanguage": top_language,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_test, probabilities)),
        "TrainingSeconds": float(training_seconds),
        "TrainingContainsHeldOutRepository": False,
    }


def _in_distribution_reference() -> dict[str, object]:
    """Evaluate the frozen RF on the original capped random production split."""
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    preprocessing = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
    frozen_model = joblib.load(ARTIFACTS_DIR / "random_forest.joblib")
    x_train, x_test, _y_train, y_test = _recreate_split(DATA_PATH, FEATURE_COLS, encoders)
    probabilities = frozen_model.predict_proba(preprocessing["imputer"].transform(x_test))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    return {
        "Evaluation": "In-distribution random reference",
        "Repository": "All repositories (interleaved)",
        "TrainRows": len(x_train),
        "TestRows": len(x_test),
        "TestFailureRate": float(y_test.mean()),
        "TopTestLanguage": "Mixed",
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_test, probabilities)),
        "TrainingSeconds": 0.0,
        "TrainingContainsHeldOutRepository": False,
    }


def _write_analysis(results: pd.DataFrame, repository_counts: pd.Series) -> None:
    held_out = results.loc[results["Evaluation"].str.startswith("Held-out:")]
    largest_share = float(repository_counts.iloc[0] / repository_counts.sum())
    top_three_share = float(repository_counts.head(HOLDOUT_REPOSITORY_COUNT).sum() / repository_counts.sum())
    table = [
        "| Evaluation | Test builds | Test failure rate | Precision | Recall | F1 | AUC | Top test language |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results.itertuples(index=False):
        table.append(
            f"| {row.Evaluation} | {row.TestRows:,} | {row.TestFailureRate:.4f} | {row.Precision:.4f} | "
            f"{row.Recall:.4f} | {row.F1:.4f} | {row.AUC:.4f} | {row.TopTestLanguage} |"
        )
    f1_range = f"{held_out.F1.min():.4f}–{held_out.F1.max():.4f}"
    ANALYSIS_PATH.write_text(
        "# Cross-Repository Generalization Evaluation\n\n"
        f"The raw field `{REPOSITORY_COLUMN}` identifies **{len(repository_counts):,}** repositories across "
        f"**{repository_counts.sum():,}** builds. Repository sizes are highly imbalanced: minimum={repository_counts.min():,}, "
        f"median={repository_counts.median():,.0f}, maximum={repository_counts.max():,}; the largest repository accounts for "
        f"{largest_share:.2%} of builds and the three largest account for {top_three_share:.2%}.\n\n"
        f"The three largest repositories were held out one at a time using a predeclared build-count rule: "
        + ", ".join(f"`{name}` ({count:,})" for name, count in repository_counts.head(HOLDOUT_REPOSITORY_COUNT).items())
        + ". This avoids selecting repositories based on favourable performance. `gh_project_name` is not in `FEATURE_COLS` and was removed before feature engineering, so no repository identity enters a model matrix.\n\n"
        + "\n".join(table)
        + "\n\n"
        + f"Held-out F1 ranges from {f1_range}. Compare these values with the in-distribution random reference in the table rather than interpreting any single held-out result in isolation. "
        + "Differences can reflect repository-specific failure base rates, language mix, team practices, and feature-distribution shift. "
        + "This experiment concerns only the selected large repositories in this TravisTorrent export and does not establish universal cross-project generalization.\n",
        encoding="utf-8",
    )


def run_cross_repo_evaluation(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Run three deterministic leave-one-repository-out research evaluations."""
    assert_features_available_for_prediction(FEATURE_COLS)
    if REPOSITORY_COLUMN in FEATURE_COLS:
        raise RuntimeError("Repository identifier must never be part of the predictive schema.")
    raw = pd.read_parquet(data_path, columns=INPUT_COLUMNS)
    repository_counts = raw[REPOSITORY_COLUMN].value_counts()
    if len(repository_counts) < HOLDOUT_REPOSITORY_COUNT:
        raise ValueError("Not enough repositories for the requested hold-out experiment.")
    holdouts = repository_counts.head(HOLDOUT_REPOSITORY_COUNT).index.tolist()

    rows = [_in_distribution_reference()]
    for repository in holdouts:
        rows.append(_evaluate_holdout(repository, raw))
        gc.collect()
    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    _write_analysis(results, repository_counts)
    print(results.to_string(index=False))
    print(f"Saved research-only outputs to {RESULTS_PATH} and {ANALYSIS_PATH}")
    return results


def main() -> None:
    run_cross_repo_evaluation()


if __name__ == "__main__":
    main()
