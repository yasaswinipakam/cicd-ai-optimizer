"""Leakage-safe temporal and repository-aware evaluation on GADFPD.

Research only: this module reads the separately prepared GitHub Actions parquet
and writes reports. It does not load, overwrite, or select production artifacts.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "gadfpd_github_actions_research.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEMPORAL_RESULTS = REPORTS_DIR / "gadfpd_temporal_baseline_results.csv"
TEMPORAL_ANALYSIS = REPORTS_DIR / "gadfpd_temporal_baseline_analysis.md"
REPO_RESULTS = REPORTS_DIR / "gadfpd_repository_holdout_results.csv"
REPO_ANALYSIS = REPORTS_DIR / "gadfpd_repository_holdout_analysis.md"

TARGET = "target_failed"
TIME = "created_at"
REPOSITORY = "repo"
NUMERIC_FEATURES = [
    "run_number", "msg_len", "is_merge", "num_parents", "additions", "deletions",
    "total_churn", "files_modified", "time_since_last_commit", "hour", "day_of_week",
    "month", "is_weekend", "is_night",
]
CATEGORICAL_FEATURES = ["event", "head_branch"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
RANDOM_STATE = 42
MAX_REPOSITORY_HOLDOUTS = 5


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _model(name: str):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def _evaluate(name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit only on supplied training rows and return threshold-0.5 metrics."""
    pipeline = Pipeline([("preprocessing", _preprocessor()), ("model", _model(name))])
    started = time.perf_counter()
    pipeline.fit(train[FEATURES], train[TARGET])
    fit_seconds = time.perf_counter() - started
    probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test[TARGET], predicted, average="binary", zero_division=0
    )
    return {
        "Model": name,
        "TrainRows": len(train),
        "TestRows": len(test),
        "TestFailureRate": float(test[TARGET].mean()),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(test[TARGET], probabilities)) if test[TARGET].nunique() == 2 else np.nan,
        "AlertRate": float(predicted.mean()),
        "TrainingSeconds": float(fit_seconds),
    }


def _load() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.data.gadfpd_pipeline` before this evaluation.")
    data = pd.read_parquet(DATA_PATH, columns=[REPOSITORY, TIME, TARGET, "global_temporal_split", *FEATURES])
    data[TIME] = pd.to_datetime(data[TIME], utc=True, errors="raise")
    return data.sort_values(TIME, kind="mergesort")


def _select_repository_holdouts(data: pd.DataFrame, cutoff: pd.Timestamp) -> list[str]:
    later = data.loc[data[TIME].gt(cutoff)]
    eligible = (
        later.groupby(REPOSITORY)[TARGET]
        .agg(test_builds="size", test_failures="sum")
        .query("test_builds >= 200 and test_failures >= 15")
        .sort_values(["test_builds", "test_failures"], ascending=False)
    )
    # Fixed, transparent selection: the five largest eligible later-period
    # repositories. All choices are reported, rather than selected by outcome.
    return eligible.head(MAX_REPOSITORY_HOLDOUTS).index.tolist()


def run_gadfpd_research_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write independent temporal and unseen-repository evaluation reports."""
    data = _load()
    train = data.loc[data["global_temporal_split"].eq("train")].copy()
    test = data.loc[data["global_temporal_split"].eq("test")].copy()
    if test[TIME].min() < train[TIME].max():
        raise RuntimeError("GADFPD temporal test overlaps training timestamps.")

    temporal = pd.DataFrame([_evaluate(name, train, test) for name in ("majority", "logistic_regression", "random_forest")])
    cutoff = train[TIME].max()
    holdout_repositories = _select_repository_holdouts(data, cutoff)
    repository_rows: list[dict] = []
    for repository in holdout_repositories:
        # Strict on both dimensions: all held-repository rows are absent from
        # training, and other-repository training rows end before final test.
        repository_train = data.loc[data[TIME].le(cutoff) & data[REPOSITORY].ne(repository)]
        repository_test = data.loc[data[TIME].gt(cutoff) & data[REPOSITORY].eq(repository)]
        if repository_test.empty or repository_test[TARGET].nunique() < 2:
            continue
        row = _evaluate("random_forest", repository_train, repository_test)
        row["Repository"] = repository
        row["TrainEnd"] = str(repository_train[TIME].max())
        row["TestStart"] = str(repository_test[TIME].min())
        repository_rows.append(row)
    repository_results = pd.DataFrame(repository_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temporal.to_csv(TEMPORAL_RESULTS, index=False)
    repository_results.to_csv(REPO_RESULTS, index=False)
    temporal_table = temporal.to_markdown(index=False, floatfmt=".4f")
    TEMPORAL_ANALYSIS.write_text(
        "# GADFPD Global Temporal Baseline\n\n"
        "This is a separate GitHub Actions research experiment. It uses only the 16 documented pre-run fields from the modern dataset; repository, workflow identifier, conclusion, duration, status, and post-run timestamps are excluded from the feature matrix. Models train on older rows and test only on the later global 20%. The 0.50 decision threshold is a baseline, not an operational alert recommendation.\n\n"
        + temporal_table
        + "\n\nDo not compare these scores directly with the deployed TravisTorrent model: platform, era, label prevalence, and features differ.\n",
        encoding="utf-8",
    )
    if repository_results.empty:
        repo_text = "# GADFPD Temporal Unseen-Repository Evaluation\n\nNo repository met the predefined later-test eligibility criteria.\n"
    else:
        repo_text = (
            "# GADFPD Temporal Unseen-Repository Evaluation\n\n"
            "Each row holds out one entire repository from training and tests only its later builds after the global temporal cutoff. The five largest repositories satisfying at least 200 later builds and 15 later failures were selected before fitting; this is a bounded, transparent study rather than performance cherry-picking.\n\n"
            + repository_results.to_markdown(index=False, floatfmt=".4f")
            + "\n\nUneven results are evidence against a single universal blocking policy; future work should use repository-specific data and calibration.\n"
        )
    REPO_ANALYSIS.write_text(repo_text, encoding="utf-8")
    return temporal, repository_results


def main() -> None:
    temporal, repositories = run_gadfpd_research_evaluation()
    print("Global temporal results:\n", temporal.to_string(index=False))
    print("\nRepository holdout results:\n", repositories.to_string(index=False))
    print(f"Reports: {TEMPORAL_RESULTS}, {REPO_RESULTS}")


if __name__ == "__main__":
    main()
