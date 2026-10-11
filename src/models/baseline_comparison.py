"""Research-only comparison of leak-free CI failure-prediction baselines.

This script reads frozen production artifacts only to recreate their strict
feature transformation and deterministic split. It never writes under
``artifacts/`` and does not change deployment model selection.
"""

from __future__ import annotations

from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.schema import FEATURE_COLS
from src.models.evaluate import _recreate_split
from src.models.train import ARTIFACTS_DIR, DATA_PATH, RANDOM_STATE


REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
RESULTS_PATH = REPORTS_DIR / "baseline_comparison_results.csv"
SUMMARY_PATH = REPORTS_DIR / "baseline_comparison.md"
CHANGE_SIZE_FEATURES = ["test_to_src_ratio", "total_files_changed", "total_test_changes"]


def _metrics(name: str, y_test, predictions, probabilities, training_seconds: float, note: str) -> dict:
    """Return one JSON/CSV-safe comparison result."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    return {
        "Model": name,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_test, probabilities)),
        "TrainingSeconds": float(training_seconds),
        "Note": note,
    }


def _fit_and_score(name: str, model, x_train, y_train, x_test, y_test, note: str) -> dict:
    started = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    return _metrics(name, y_test, model.predict(x_test), model.predict_proba(x_test)[:, 1], elapsed, note)


def _write_summary(results: pd.DataFrame) -> None:
    """Write a compact, number-led Markdown report without deployment claims."""
    columns = ["Model", "Precision", "Recall", "F1", "AUC", "TrainingSeconds", "Note"]
    table_rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in results[columns].itertuples(index=False):
        model, precision, recall, f1, auc, training_seconds, note = row
        table_rows.append(
            f"| {model} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {auc:.4f} | "
            f"{training_seconds:.2f} | {note} |"
        )
    table = "\n".join(table_rows)
    rf = results.loc[results["Model"] == "Random Forest", "F1"].iloc[0]
    mlp = results.loc[results["Model"] == "Tabular MLP", "F1"].iloc[0]
    change = results.loc[results["Model"] == "Change-size Logistic Regression", "F1"].iloc[0]
    SUMMARY_PATH.write_text(
        "# Strict Pre-Build Baseline and Model Comparison\n\n"
        "All rows use the frozen pipeline's availability-guarded `FEATURE_COLS`, "
        "200,000-row cap, and deterministic 80/20 stratified split (`random_state=42`). "
        "These are research comparisons only; no row changes deployment selection.\n\n"
        + table
        + "\n\n"
        + f"Random Forest F1 is {rf:.4f}; the change-size-only logistic baseline is {change:.4f}. "
        + f"The tabular MLP F1 is {mlp:.4f}.\n\n"
        + "The raw previous-status field has no `failed` category in the saved encoder "
        + "(`build_found`, `merge_found`, `no_previous_build`), so the requested "
        + "previous-build-failed heuristic cannot identify a prior failure and predicts passed for every row.\n",
        encoding="utf-8",
    )


def run_comparison(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Run research-only baselines on the production-equivalent strict split."""
    assert_features_available_for_prediction(FEATURE_COLS)
    preprocessing = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    feature_columns = preprocessing["feature_cols"]
    if feature_columns != FEATURE_COLS:
        raise ValueError("Frozen preprocessing schema does not match FEATURE_COLS.")

    x_train, x_test, y_train, y_test = _recreate_split(data_path, feature_columns, encoders)
    imputer = preprocessing["imputer"]
    x_train_full = imputer.transform(x_train)
    x_test_full = imputer.transform(x_test)
    feature_index = {column: index for index, column in enumerate(feature_columns)}
    change_indices = [feature_index[column] for column in CHANGE_SIZE_FEATURES]
    x_train_change = x_train_full[:, change_indices]
    x_test_change = x_test_full[:, change_indices]
    positive_rate = float(y_train.mean())
    majority_label = int(y_train.mode().iloc[0])
    results: list[dict] = []

    majority_predictions = pd.Series(majority_label, index=y_test.index).to_numpy()
    majority_probabilities = pd.Series(positive_rate, index=y_test.index).to_numpy()
    results.append(
        _metrics(
            "Majority Class",
            y_test,
            majority_predictions,
            majority_probabilities,
            0.0,
            "Always predicts the training-set majority class.",
        )
    )

    previous_encoder = encoders["prev_status_enc"]
    if "failed" in previous_encoder.classes_:
        failed_code = int(previous_encoder.transform(["failed"])[0])
        previous_predictions = x_test["prev_status_enc"].eq(failed_code).astype(int).to_numpy()
        previous_note = "Predicts failure only when the previous-status encoding is failed."
    else:
        previous_predictions = pd.Series(0, index=y_test.index).to_numpy()
        previous_note = "No 'failed' category exists in this dataset's previous-status encoding; predicts passed."
    results.append(
        _metrics(
            "Previous-Status Heuristic",
            y_test,
            previous_predictions,
            previous_predictions.astype(float),
            0.0,
            previous_note,
        )
    )

    change_scaler = StandardScaler()
    results.append(
        _fit_and_score(
            "Change-size Logistic Regression",
            LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=RANDOM_STATE),
            change_scaler.fit_transform(x_train_change),
            y_train,
            change_scaler.transform(x_test_change),
            y_test,
            "Uses only strict derived change-size features.",
        )
    )

    full_scaler = StandardScaler()
    results.append(
        _fit_and_score(
            "Full Logistic Regression",
            LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=RANDOM_STATE),
            full_scaler.fit_transform(x_train_full),
            y_train,
            full_scaler.transform(x_test_full),
            y_test,
            "Linear baseline over all strict pre-build features.",
        )
    )

    results.append(
        _fit_and_score(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            x_train_full,
            y_train,
            x_test_full,
            y_test,
            "Same hyperparameters as the frozen production model; research-only refit.",
        )
    )

    class_counts = y_train.value_counts()
    results.append(
        _fit_and_score(
            "XGBoost",
            XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=float(class_counts[0] / class_counts[1]),
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
            ),
            x_train_full,
            y_train,
            x_test_full,
            y_test,
            "Same hyperparameters as the prior XGBoost comparison; research-only refit.",
        )
    )

    mlp_scaler = StandardScaler()
    results.append(
        _fit_and_score(
            "Tabular MLP",
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=30,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                random_state=RANDOM_STATE,
            ),
            mlp_scaler.fit_transform(x_train_full),
            y_train,
            mlp_scaler.transform(x_test_full),
            y_test,
            "Single-hidden-layer tabular comparison; no sequence data or tuning.",
        )
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(results)
    comparison.to_csv(RESULTS_PATH, index=False)
    _write_summary(comparison)
    print(comparison.to_string(index=False))
    print(f"Saved research-only results to {RESULTS_PATH}")
    return comparison


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
