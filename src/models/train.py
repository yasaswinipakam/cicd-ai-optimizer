"""Train CI/CD failure models in explicit fast or full chronological modes.

``fast`` preserves the 200k-row random-split comparison workflow for development.
``full`` is the deployable workflow: it trains a Random Forest on the earliest
80% of builds and reports only on the later 20%, with no shuffled future rows.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.engineer import engineer_features
from src.features.schema import FEATURE_COLS, FEATURE_SOURCE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
MAX_TRAINING_ROWS = 200_000
MAX_SVM_TRAINING_ROWS = 20_000
TIMESTAMP_COLUMN = "gh_build_started_at"
TARGET_COLUMN = "tr_status"
FULL_TRAIN_FRACTION = 0.80
VALIDATION_FRACTION_OF_TRAIN = 0.10
MAX_ALERT_RATE = 0.40
THRESHOLDS = [round(value / 100, 2) for value in range(5, 100, 5)]


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


def _load_input(data_path: Path) -> pd.DataFrame:
    """Load only the raw/processed columns needed for strict model features."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. Run `python -m src.data.run_pipeline` first."
        )
    columns = list(dict.fromkeys([TARGET_COLUMN, *FEATURE_SOURCE_COLUMNS]))
    return pd.read_parquet(data_path, columns=columns, engine="pyarrow")


def _engineer_split(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Fit categorical encoders/imputation only on training rows, then transform test."""
    encoders: dict = {}
    train, feature_columns = engineer_features(
        train_raw, encoders=encoders, fit_encoders=True, save_encoders=False
    )
    test, transformed_columns = engineer_features(
        test_raw, encoders=encoders, fit_encoders=False, save_encoders=False
    )
    if feature_columns != FEATURE_COLS or transformed_columns != FEATURE_COLS:
        raise RuntimeError("Feature engineering did not return the canonical schema.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURE_COLS])
    x_test = imputer.transform(test[FEATURE_COLS])
    y_train = train[TARGET_COLUMN].eq("failed").astype(int)
    y_test = test[TARGET_COLUMN].eq("failed").astype(int)
    return x_train, x_test, y_train, y_test, imputer, encoders


def _metrics(y_true, probabilities, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_true, probabilities)),
        "AlertRate": float(predictions.mean()),
        "FailureCaptureRate": float(recall),
    }


def _select_warning_threshold(y_validation, probabilities) -> tuple[float, dict[str, float | str]]:
    """Choose a developer-warning threshold with bounded alert volume.

    The policy maximizes failure capture while allowing alerts on no more than
    40% of validation builds and requiring precision no worse than the observed
    validation failure rate. This makes the operational objective explicit rather
    than claiming that F1 alone dictates the alert policy.
    """
    base_failure_rate = float(pd.Series(y_validation).mean())
    candidates: list[tuple[float, float, float]] = []
    for threshold in THRESHOLDS:
        values = _metrics(y_validation, probabilities, threshold)
        if values["AlertRate"] <= MAX_ALERT_RATE and values["Precision"] >= base_failure_rate:
            candidates.append((values["Recall"], values["F1"], threshold))
    if candidates:
        _recall, _f1, threshold = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        selection = "maximize recall subject to alert-rate and precision floors"
    else:
        _f1, threshold = max(
            ((_metrics(y_validation, probabilities, value)["F1"], value) for value in THRESHOLDS),
            key=lambda item: (item[0], item[1]),
        )
        selection = "F1 fallback because no policy-constrained threshold was available"
    chosen = _metrics(y_validation, probabilities, threshold)
    return float(threshold), {
        "name": "developer_warning_policy",
        "selection_rule": selection,
        "max_alert_rate": MAX_ALERT_RATE,
        "minimum_precision": base_failure_rate,
        "validation_precision": chosen["Precision"],
        "validation_recall": chosen["Recall"],
        "validation_alert_rate": chosen["AlertRate"],
    }


def _persist_deployable(
    model: object,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    encoders: dict,
    *,
    mode: str,
    threshold: float,
    threshold_policy: dict,
    metadata: dict,
) -> None:
    """Save the artifacts consumed by live inference with schema metadata."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "random_forest.joblib")
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")
    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "feature_cols": FEATURE_COLS.copy(),
            "decision_threshold": threshold,
            "training_mode": mode,
            "threshold_policy": threshold_policy,
            "metadata": metadata,
        },
        ARTIFACTS_DIR / "preprocessing.joblib",
    )
    (ARTIFACTS_DIR / "best_model.txt").write_text("random_forest\n", encoding="utf-8")


def _fit_fast_models(data_path: Path) -> dict[str, object]:
    """Fit the historical 200k random-split comparison set for development."""
    raw = _load_input(data_path)
    if len(raw) > MAX_TRAINING_ROWS:
        raw = raw.sample(n=MAX_TRAINING_ROWS, random_state=RANDOM_STATE)
    target = raw[TARGET_COLUMN].eq("failed").astype(int)
    train_raw, test_raw = train_test_split(
        raw, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )
    x_train, _x_test, y_train, _y_test, imputer, encoders = _engineer_split(train_raw, test_raw)
    scaler = StandardScaler().fit(x_train)
    class_counts = y_train.value_counts()
    scale_pos_weight = float(class_counts[0] / class_counts[1])
    models: dict[str, object] = {
        "random_forest": _random_forest(),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "svm": SVC(
            kernel="rbf", C=10, gamma="scale", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
        ),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        started = time.perf_counter()
        if name == "svm":
            svm_rows = min(MAX_SVM_TRAINING_ROWS, len(x_train))
            model.fit(scaler.transform(x_train[:svm_rows]), y_train.iloc[:svm_rows])
        else:
            model.fit(x_train, y_train)
        joblib.dump(model, ARTIFACTS_DIR / f"{name}.joblib")
        print(f"{name}: trained in {time.perf_counter() - started:.2f}s")
    _persist_deployable(
        models["random_forest"], imputer, scaler, encoders,
        mode="fast", threshold=0.5,
        threshold_policy={"name": "default", "selection_rule": "fast development mode uses 0.50"},
        metadata={"sample_rows": len(raw), "split": "stratified_random", "test_rows": len(test_raw)},
    )
    print("Fast mode complete: four comparison models persisted; RF remains the live model.")
    return models


def _write_full_report(result: dict[str, object]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(REPORTS_DIR / "enhanced_full_temporal_evaluation.csv", index=False)
    (REPORTS_DIR / "enhanced_full_temporal_evaluation.md").write_text(
        "# Enhanced Full Temporal Evaluation\n\n"
        "The deployable Random Forest uses the expanded availability-guarded pre-build feature schema. "
        "It was fitted on the earliest 80% of builds; its decision threshold was selected on a chronological validation "
        "window inside that training period, and the final 20% remained untouched until evaluation.\n\n"
        + "| Train rows | Test rows | Threshold | Precision | Recall | F1 | AUC | Alert rate | Failure capture rate |\n"
        + "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + f"| {result['TrainRows']:,} | {result['TestRows']:,} | {result['DecisionThreshold']:.2f} | "
        + f"{result['Precision']:.4f} | {result['Recall']:.4f} | {result['F1']:.4f} | {result['AUC']:.4f} | "
        + f"{result['AlertRate']:.4f} | {result['FailureCaptureRate']:.4f} |\n\n"
        + f"Train range: {result['TrainStart']} to {result['TrainEnd']}. Test range: {result['TestStart']} to {result['TestEnd']}. "
        + "This result is a chronological within-dataset evaluation, not proof of cross-repository generalization.\n",
        encoding="utf-8",
    )


def _fit_full_temporal(data_path: Path) -> dict[str, object]:
    """Fit the deployable RF on early history and evaluate only later history."""
    raw = _load_input(data_path)
    timestamps = pd.to_datetime(raw[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Full mode requires parseable gh_build_started_at timestamps.")
    ordered = raw.assign(**{TIMESTAMP_COLUMN: timestamps}).sort_values(TIMESTAMP_COLUMN, kind="mergesort")
    split_index = int(len(ordered) * FULL_TRAIN_FRACTION)
    train_raw = ordered.iloc[:split_index].copy()
    test_raw = ordered.iloc[split_index:].copy()
    if test_raw[TIMESTAMP_COLUMN].min() < train_raw[TIMESTAMP_COLUMN].max():
        raise RuntimeError("Chronological split leakage: test rows precede training rows.")

    validation_index = int(len(train_raw) * (1 - VALIDATION_FRACTION_OF_TRAIN))
    selection_train_raw = train_raw.iloc[:validation_index].copy()
    validation_raw = train_raw.iloc[validation_index:].copy()
    validation_rows = len(validation_raw)
    x_selection, x_validation, y_selection, y_validation, _sel_imputer, _sel_encoders = _engineer_split(
        selection_train_raw, validation_raw
    )
    selector_model = _random_forest().fit(x_selection, y_selection)
    threshold, policy = _select_warning_threshold(
        y_validation, selector_model.predict_proba(x_validation)[:, 1]
    )
    del selection_train_raw, validation_raw, x_selection, x_validation, y_selection, y_validation, _sel_imputer, _sel_encoders, selector_model
    gc.collect()

    x_train, x_test, y_train, y_test, imputer, encoders = _engineer_split(train_raw, test_raw)
    model = _random_forest()
    started = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = _metrics(y_test, probabilities, threshold)
    scaler = StandardScaler().fit(x_train)
    result: dict[str, object] = {
        "Mode": "full",
        "TrainRows": len(train_raw),
        "TestRows": len(test_raw),
        "ValidationRows": validation_rows,
        "TrainStart": str(train_raw[TIMESTAMP_COLUMN].min()),
        "TrainEnd": str(train_raw[TIMESTAMP_COLUMN].max()),
        "TestStart": str(test_raw[TIMESTAMP_COLUMN].min()),
        "TestEnd": str(test_raw[TIMESTAMP_COLUMN].max()),
        "DecisionThreshold": threshold,
        "TrainingSeconds": training_seconds,
        **metrics,
    }
    _persist_deployable(
        model, imputer, scaler, encoders,
        mode="full", threshold=threshold, threshold_policy=policy, metadata=result,
    )
    _write_full_report(result)
    print(pd.DataFrame([result]).to_string(index=False))
    print("Full temporal mode complete: deployable RF and preprocessing artifacts updated.")
    return {"random_forest": model}


def train_models(data_path: Path = DATA_PATH, mode: str = "fast") -> dict[str, object]:
    """Train models in explicit ``fast`` or deployable ``full`` mode."""
    assert_features_available_for_prediction(FEATURE_COLS)
    if mode == "fast":
        return _fit_fast_models(data_path)
    if mode == "full":
        return _fit_full_temporal(data_path)
    raise ValueError("mode must be 'fast' or 'full'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CI/CD failure models.")
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    args = parser.parse_args()
    train_models(mode=args.mode)


if __name__ == "__main__":
    main()
