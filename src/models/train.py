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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.engineer import HISTORY_AUX_COLS, add_rolling_history, engineer_features, training_failure_prior
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
    """Load columns needed for strict model features, repo identity, and history aux columns."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. Run `python -m src.data.run_pipeline` first."
        )
    import pyarrow.parquet as pq
    schema_cols = set(pq.read_schema(data_path).names)
    columns = list(dict.fromkeys(
        c for c in [TARGET_COLUMN, "gh_project_name", *FEATURE_SOURCE_COLUMNS, *HISTORY_AUX_COLS]
        if c in schema_cols
    ))
    return pd.read_parquet(data_path, columns=columns, engine="pyarrow")


def _engineer_split(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """Fit encoders, imputation and the cold-start prior only on training rows."""
    encoders: dict = {}
    prior = training_failure_prior(train_raw, label_col=TARGET_COLUMN)
    train, feature_columns = engineer_features(
        train_raw, encoders=encoders, fit_encoders=True, save_encoders=False, cold_start_prior=prior
    )
    test, transformed_columns = engineer_features(
        test_raw, encoders=encoders, fit_encoders=False, save_encoders=False, cold_start_prior=prior
    )
    if feature_columns != FEATURE_COLS or transformed_columns != FEATURE_COLS:
        raise RuntimeError("Feature engineering did not return the canonical schema.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[FEATURE_COLS])
    x_test = imputer.transform(test[FEATURE_COLS])
    y_train = train[TARGET_COLUMN].eq("failed").astype(int)
    y_test = test[TARGET_COLUMN].eq("failed").astype(int)
    return x_train, x_test, y_train, y_test, imputer, encoders, prior


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
    cold_start_prior: float,
    model_name: str = "random_forest",
) -> None:
    """Save the artifacts consumed by live inference with schema metadata."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / f"{model_name}.joblib")
    joblib.dump(encoders, ARTIFACTS_DIR / "encoders.joblib")
    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "feature_cols": FEATURE_COLS.copy(),
            "cold_start_prior": cold_start_prior,
            "decision_threshold": threshold,
            "training_mode": mode,
            "threshold_policy": threshold_policy,
            "metadata": metadata,
        },
        ARTIFACTS_DIR / "preprocessing.joblib",
    )
    (ARTIFACTS_DIR / "best_model.txt").write_text(f"{model_name}\n", encoding="utf-8")


def _fit_fast_models(data_path: Path) -> dict[str, object]:
    """Fit the historical 200k random-split comparison set for development."""
    raw = _load_input(data_path)
    # Sort chronologically so rolling history windows are correct, then sample.
    raw = raw.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
    raw = add_rolling_history(raw, label_col=TARGET_COLUMN)
    if len(raw) > MAX_TRAINING_ROWS:
        raw = raw.sample(n=MAX_TRAINING_ROWS, random_state=RANDOM_STATE)
    target = raw[TARGET_COLUMN].eq("failed").astype(int)
    train_raw, test_raw = train_test_split(
        raw, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )
    x_train, _x_test, y_train, _y_test, imputer, encoders, prior = _engineer_split(train_raw, test_raw)
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
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
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
        cold_start_prior=prior,
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
    # Rolling history must be computed on the full sorted dataset before splitting
    # so that training rows' windows never reach into the held-out test period.
    ordered = add_rolling_history(ordered, label_col=TARGET_COLUMN)
    split_index = int(len(ordered) * FULL_TRAIN_FRACTION)
    train_raw = ordered.iloc[:split_index].copy()
    test_raw = ordered.iloc[split_index:].copy()
    if test_raw[TIMESTAMP_COLUMN].min() < train_raw[TIMESTAMP_COLUMN].max():
        raise RuntimeError("Chronological split leakage: test rows precede training rows.")

    validation_index = int(len(train_raw) * (1 - VALIDATION_FRACTION_OF_TRAIN))
    selection_train_raw = train_raw.iloc[:validation_index].copy()
    validation_raw = train_raw.iloc[validation_index:].copy()
    validation_rows = len(validation_raw)
    x_selection, x_validation, y_selection, y_validation, _sel_imputer, _sel_encoders, _sel_prior = _engineer_split(
        selection_train_raw, validation_raw
    )
    selector_model = _random_forest().fit(x_selection, y_selection)
    threshold, policy = _select_warning_threshold(
        y_validation, selector_model.predict_proba(x_validation)[:, 1]
    )
    del selection_train_raw, validation_raw, x_selection, x_validation, y_selection, y_validation, _sel_imputer, _sel_encoders, selector_model
    gc.collect()

    x_train, x_test, y_train, y_test, imputer, encoders, prior = _engineer_split(train_raw, test_raw)
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
        cold_start_prior=prior,
    )
    _write_full_report(result)
    print(pd.DataFrame([result]).to_string(index=False))
    print("Full temporal mode complete: deployable RF and preprocessing artifacts updated.")
    return {"random_forest": model}


_RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.2, 0.3],
    "class_weight": ["balanced", "balanced_subsample"],
}

_XGB_PARAM_DIST = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
}

_LGBM_PARAM_DIST = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [6, 8, 10, -1],
    "num_leaves": [31, 63, 127, 255],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_samples": [10, 20, 50],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 0.1, 1.0],
}


def _tune_fast_models(data_path: Path) -> dict[str, object]:
    """Hyperparameter search on a randomly-sampled 200k dev split.

    WARNING — development reference only.  The 200k subset is drawn by random
    sample, so chronological order is not preserved after sampling.  The CV
    folds here (StratifiedKFold, shuffle=False, applied to the already-random
    sample) do NOT constitute a valid chronological evaluation.  Selected
    hyperparameters are starting points for --mode full, not performance claims.
    """
    raw = _load_input(data_path)
    raw = raw.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
    raw = add_rolling_history(raw, label_col=TARGET_COLUMN)
    if len(raw) > MAX_TRAINING_ROWS:
        raw = raw.sample(n=MAX_TRAINING_ROWS, random_state=RANDOM_STATE)

    target = raw[TARGET_COLUMN].eq("failed").astype(int)
    train_raw, test_raw = train_test_split(
        raw, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )
    x_train, x_test, y_train, y_test, imputer, encoders, prior = _engineer_split(train_raw, test_raw)

    class_counts = y_train.value_counts()
    scale_pos_weight = float(class_counts[0] / class_counts[1])
    # shuffle=False: do not actively mix future/past rows on top of the already-
    # random sample. Still not chronological — see function docstring.
    cv = StratifiedKFold(n_splits=3, shuffle=False)

    print("Tuning Random Forest (n_iter=25, cv=3, scoring=f1) ...")
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_base, _RF_PARAM_DIST, n_iter=25, scoring="f1", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    rf_search.fit(x_train, y_train)
    print(f"  Best RF params: {rf_search.best_params_}")
    print(f"  Best RF CV F1:  {rf_search.best_score_:.4f}")

    print("Tuning XGBoost (n_iter=25, cv=3, scoring=f1) ...")
    # Boosters run single-threaded inside the parallel search: nesting OpenMP
    # pools inside loky workers oversubscribes cores and can deadlock.
    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=1, tree_method="hist",
    )
    xgb_search = RandomizedSearchCV(
        xgb_base, _XGB_PARAM_DIST, n_iter=25, scoring="f1", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    xgb_search.fit(x_train, y_train)
    print(f"  Best XGB params: {xgb_search.best_params_}")
    print(f"  Best XGB CV F1:  {xgb_search.best_score_:.4f}")

    print("Tuning LightGBM (n_iter=25, cv=3, scoring=f1) ...")
    lgbm_base = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
    )
    lgbm_search = RandomizedSearchCV(
        lgbm_base, _LGBM_PARAM_DIST, n_iter=25, scoring="f1", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    lgbm_search.fit(x_train, y_train)
    print(f"  Best LGBM params: {lgbm_search.best_params_}")
    print(f"  Best LGBM CV F1:  {lgbm_search.best_score_:.4f}")

    # DT and SVM keep their existing params for the comparison table.
    scaler = StandardScaler().fit(x_train)
    dt = DecisionTreeClassifier(
        max_depth=12, min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE,
    ).fit(x_train, y_train)
    svm_rows = min(MAX_SVM_TRAINING_ROWS, len(x_train))
    svm = SVC(
        kernel="rbf", C=10, gamma="scale", probability=True,
        class_weight="balanced", random_state=RANDOM_STATE,
    ).fit(scaler.transform(x_train[:svm_rows]), y_train.iloc[:svm_rows])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[str, object] = {
        "random_forest": rf_search.best_estimator_,
        "xgboost": xgb_search.best_estimator_.set_params(n_jobs=-1),
        "lightgbm": lgbm_search.best_estimator_.set_params(n_jobs=-1),
        "decision_tree": dt,
        "svm": svm,
    }
    for name, model in models.items():
        joblib.dump(model, ARTIFACTS_DIR / f"{name}.joblib")

    _persist_deployable(
        rf_search.best_estimator_, imputer, scaler, encoders,
        mode="tune",
        threshold=0.5,
        threshold_policy={"name": "default", "selection_rule": "tune mode uses 0.50"},
        metadata={
            "sample_rows": len(raw),
            "split": "stratified_random",
            "rf_best_params": rf_search.best_params_,
            "rf_best_cv_f1": rf_search.best_score_,
            "xgb_best_params": xgb_search.best_params_,
            "xgb_best_cv_f1": xgb_search.best_score_,
            "lgbm_best_params": lgbm_search.best_params_,
            "lgbm_best_cv_f1": lgbm_search.best_score_,
        },
        cold_start_prior=prior,
    )
    print("Tune mode complete: best RF, XGBoost, and LightGBM persisted.")
    return models


# Best hyperparameters from the last --mode tune run.
# Update these after re-running --mode tune on new data.
_STACK_RF_PARAMS: dict = {
    "n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 2,
    "max_features": "sqrt", "max_depth": None, "class_weight": "balanced",
}
_STACK_XGB_PARAMS: dict = {
    "n_estimators": 400, "max_depth": 10, "learning_rate": 0.2,
    "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 1,
    "gamma": 0.1, "reg_alpha": 0, "reg_lambda": 2.0,
}
_STACK_LGBM_PARAMS: dict = {
    "n_estimators": 300, "max_depth": 10, "learning_rate": 0.2,
    "num_leaves": 255, "subsample": 0.9, "colsample_bytree": 0.9,
    "min_child_samples": 20, "reg_alpha": 1.0, "reg_lambda": 1.0,
}


def _fit_stacking_models(data_path: Path) -> dict[str, object]:
    """Stacked ensemble: RF + XGB + LGBM base learners → LR meta-learner (cv=5)."""
    raw = _load_input(data_path)
    raw = raw.sort_values(TIMESTAMP_COLUMN, kind="mergesort")
    raw = add_rolling_history(raw, label_col=TARGET_COLUMN)
    if len(raw) > MAX_TRAINING_ROWS:
        raw = raw.sample(n=MAX_TRAINING_ROWS, random_state=RANDOM_STATE)

    target = raw[TARGET_COLUMN].eq("failed").astype(int)
    train_raw, test_raw = train_test_split(
        raw, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )
    x_train, x_test, y_train, y_test, imputer, encoders, prior = _engineer_split(train_raw, test_raw)

    class_counts = y_train.value_counts()
    scale_pos_weight = float(class_counts[0] / class_counts[1])

    estimators = [
        ("rf", RandomForestClassifier(
            **_STACK_RF_PARAMS, random_state=RANDOM_STATE, n_jobs=-1,
        )),
        ("xgb", XGBClassifier(
            **_STACK_XGB_PARAMS, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
        )),
        ("lgbm", LGBMClassifier(
            **_STACK_LGBM_PARAMS, scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        # Base models already use every core; parallel folds would nest OpenMP pools.
        n_jobs=None,
    )
    started = time.perf_counter()
    print("Fitting StackingClassifier (cv=5, 3 base models + LR meta) ...")
    stack.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    print(f"  Stacking: trained in {elapsed:.1f}s")

    proba = stack.predict_proba(x_test)[:, 1]
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, proba)
    test_f1 = f1_score(y_test, (proba >= 0.5).astype(int))
    print(f"  Stacking test F1={test_f1:.4f}, AUC={test_auc:.4f}")

    scaler = StandardScaler().fit(x_train)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(stack, ARTIFACTS_DIR / "stacking.joblib")
    _persist_deployable(
        stack, imputer, scaler, encoders,
        mode="stack", threshold=0.5,
        threshold_policy={"name": "default", "selection_rule": "stack mode uses 0.50"},
        metadata={"sample_rows": len(raw), "split": "stratified_random",
                  "test_f1": test_f1, "test_auc": test_auc},
        cold_start_prior=prior,
        model_name="stacking",
    )
    print("Stack mode complete: StackingClassifier persisted as live model.")
    return {"stacking": stack}


def train_models(data_path: Path = DATA_PATH, mode: str = "fast") -> dict[str, object]:
    """Train models in explicit ``fast``, ``tune``, ``stack``, or deployable ``full`` mode."""
    assert_features_available_for_prediction(FEATURE_COLS)
    if mode == "fast":
        return _fit_fast_models(data_path)
    if mode == "tune":
        return _tune_fast_models(data_path)
    if mode == "stack":
        return _fit_stacking_models(data_path)
    if mode == "full":
        return _fit_full_temporal(data_path)
    raise ValueError("mode must be 'fast', 'tune', 'stack', or 'full'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CI/CD failure models.")
    parser.add_argument("--mode", choices=("fast", "tune", "stack", "full"), default="fast")
    args = parser.parse_args()
    train_models(mode=args.mode)


if __name__ == "__main__":
    main()
