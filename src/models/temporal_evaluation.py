"""Research-only chronological evaluation of strict CI failure prediction."""

from __future__ import annotations

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
from src.features.schema import FEATURE_COLS, FEATURE_SOURCE_COLUMNS
from src.models.evaluate import _recreate_split
from src.models.train import ARTIFACTS_DIR, DATA_PATH, RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "temporal_vs_random_split_results.csv"
ANALYSIS_PATH = REPORTS_DIR / "temporal_evaluation_analysis.md"
TIMESTAMP_COLUMN = "gh_build_started_at"
TARGET_COLUMN = "tr_status"
TEMPORAL_TRAIN_FRACTION = 0.8
TEMPORAL_INPUT_COLUMNS = list(dict.fromkeys([TARGET_COLUMN, *FEATURE_SOURCE_COLUMNS]))


def _random_forest() -> RandomForestClassifier:
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
    """Fit categorical encoders only on the chronological training period."""
    sources = {
        "prev_status_enc": "git_prev_commit_resolution_status",
        "lang_enc": "gh_lang",
    }
    return {
        feature: LabelEncoder().fit(training_raw[source].fillna("__missing__").astype(str))
        for feature, source in sources.items()
    }


def _score(model, x_train, y_train, x_test, y_test, split: str, train_start, train_end, test_start, test_end) -> dict:
    """Fit one isolated model and return common comparison metadata and metrics."""
    imputer = SimpleImputer(strategy="median")
    x_train_ready = imputer.fit_transform(x_train)
    x_test_ready = imputer.transform(x_test)
    started = time.perf_counter()
    model.fit(x_train_ready, y_train)
    elapsed = time.perf_counter() - started
    predictions = model.predict(x_test_ready)
    probabilities = model.predict_proba(x_test_ready)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    return {
        "Split": split,
        "TrainRows": len(x_train),
        "TestRows": len(x_test),
        "TrainStart": str(train_start),
        "TrainEnd": str(train_end),
        "TestStart": str(test_start),
        "TestEnd": str(test_end),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_test, probabilities)),
        "TrainingSeconds": float(elapsed),
    }


def _random_split_reference(preprocessing: dict, encoders: dict) -> dict:
    """Re-run the exact capped production random-split methodology."""
    x_train, x_test, y_train, y_test = _recreate_split(DATA_PATH, FEATURE_COLS, encoders)
    # The saved imputer was fitted on this deterministic production training split.
    x_train_ready = preprocessing["imputer"].transform(x_train)
    x_test_ready = preprocessing["imputer"].transform(x_test)
    model = _random_forest()
    started = time.perf_counter()
    model.fit(x_train_ready, y_train)
    elapsed = time.perf_counter() - started
    predictions = model.predict(x_test_ready)
    probabilities = model.predict_proba(x_test_ready)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    return {
        "Split": "Random (production methodology)",
        "TrainRows": len(x_train),
        "TestRows": len(x_test),
        "TrainStart": "Interleaved 2011-2016",
        "TrainEnd": "Interleaved 2011-2016",
        "TestStart": "Interleaved 2011-2016",
        "TestEnd": "Interleaved 2011-2016",
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_test, probabilities)),
        "TrainingSeconds": float(elapsed),
    }


def _write_analysis(results: pd.DataFrame) -> None:
    random_result = results.loc[results["Split"] == "Random (production methodology)"].iloc[0]
    temporal = results.loc[results["Split"] == "Temporal (earlier to later)"].iloc[0]
    f1_difference = float(temporal["F1"] - random_result["F1"])
    auc_difference = float(temporal["AUC"] - random_result["AUC"])
    table = [
        "| Split | Train rows | Test rows | Train date range | Test date range | Precision | Recall | F1 | AUC | Training seconds |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        table.append(
            f"| {row.Split} | {row.TrainRows:,} | {row.TestRows:,} | {row.TrainStart} to {row.TrainEnd} | "
            f"{row.TestStart} to {row.TestEnd} | {row.Precision:.4f} | {row.Recall:.4f} | "
            f"{row.F1:.4f} | {row.AUC:.4f} | {row.TrainingSeconds:.2f} |"
        )
    direction = "lower" if f1_difference < 0 else "higher"
    ANALYSIS_PATH.write_text(
        "# Temporal versus Random Split Evaluation\n\n"
        "Both evaluations use the same strict pre-build `FEATURE_COLS` and frozen Random Forest hyperparameters. "
        "The random reference reproduces the production capped sampling methodology; the temporal evaluation uses all "
        "chronologically ordered builds, fitting encoders and imputation only on the earlier training period.\n\n"
        + "\n".join(table)
        + "\n\n"
        + f"On this dataset and timeframe, temporal F1 is {abs(f1_difference):.4f} points {direction} than the random-split reference "
        + f"({temporal.F1:.4f} versus {random_result.F1:.4f}); temporal AUC differs by {auc_difference:+.4f}. "
        + "This is evidence only for this TravisTorrent export and its 2011-2016 chronology; it does not establish deployment readiness or generalization to other repositories/timeframes.\n",
        encoding="utf-8",
    )


def run_temporal_evaluation() -> pd.DataFrame:
    """Fit a research-only chronological model and compare it with random split."""
    assert_features_available_for_prediction(FEATURE_COLS)
    preprocessing = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
    saved_encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    if preprocessing["feature_cols"] != FEATURE_COLS:
        raise ValueError("Frozen preprocessing schema does not match FEATURE_COLS.")

    random_result = _random_split_reference(preprocessing, saved_encoders)
    raw = pd.read_parquet(DATA_PATH, columns=TEMPORAL_INPUT_COLUMNS)
    timestamps = pd.to_datetime(raw[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Temporal evaluation blocked: gh_build_started_at contains unparsable timestamps.")
    ordered = raw.assign(**{TIMESTAMP_COLUMN: timestamps}).sort_values(TIMESTAMP_COLUMN, kind="mergesort")
    split_index = int(len(ordered) * TEMPORAL_TRAIN_FRACTION)
    temporal_train_raw = ordered.iloc[:split_index].copy()
    temporal_test_raw = ordered.iloc[split_index:].copy()
    train_end = temporal_train_raw[TIMESTAMP_COLUMN].max()
    test_start = temporal_test_raw[TIMESTAMP_COLUMN].min()
    if test_start < train_end:
        raise RuntimeError("Temporal split leakage: a test timestamp precedes the latest training timestamp.")
    if len(temporal_train_raw) + len(temporal_test_raw) != len(ordered):
        raise RuntimeError("Temporal split row counts do not sum to the source dataset.")

    temporal_encoders = _fit_encoders(temporal_train_raw)
    temporal_train, _ = engineer_features(
        temporal_train_raw,
        encoders=temporal_encoders,
        fit_encoders=False,
        save_encoders=False,
    )
    temporal_test, _ = engineer_features(
        temporal_test_raw,
        encoders=temporal_encoders,
        fit_encoders=False,
        save_encoders=False,
    )
    y_train = temporal_train[TARGET_COLUMN].eq("failed").astype(int)
    y_test = temporal_test[TARGET_COLUMN].eq("failed").astype(int)
    temporal_result = _score(
        _random_forest(),
        temporal_train[FEATURE_COLS],
        y_train,
        temporal_test[FEATURE_COLS],
        y_test,
        "Temporal (earlier to later)",
        temporal_train_raw[TIMESTAMP_COLUMN].min(),
        train_end,
        test_start,
        temporal_test_raw[TIMESTAMP_COLUMN].max(),
    )
    results = pd.DataFrame([random_result, temporal_result])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    _write_analysis(results)
    print(results.to_string(index=False))
    print(f"Saved research-only outputs to {RESULTS_PATH} and {ANALYSIS_PATH}")
    return results


def main() -> None:
    run_temporal_evaluation()


if __name__ == "__main__":
    main()
