"""Research-only leakage-regime gradient experiment for this TravisTorrent export.

The script recreates the frozen strict production split and trains isolated
Random Forests. It never writes to ``artifacts/`` or changes deployment state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import joblib

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "cicd-ai-optimizer-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.features.availability_guard import assert_features_available_for_prediction
from src.features.schema import FEATURE_COLS
from src.models.evaluate import _recreate_split
from src.models.train import ARTIFACTS_DIR, DATA_PATH, RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REGISTRY_PATH = REPORTS_DIR / "feature_availability_registry.json"
RESULTS_PATH = REPORTS_DIR / "leakage_regime_results.csv"
FIGURE_PATH = REPORTS_DIR / "leakage_regime_gradient.png"
ANALYSIS_PATH = REPORTS_DIR / "leakage_regime_analysis.md"
PRODUCTION_F1 = 0.4299660961562028
PRODUCTION_AUC = 0.6902338498060125
REPRODUCTION_TOLERANCE = 1e-6


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


def _post_build_columns() -> list[str]:
    """Select all registry-labelled log/duration fields except the target itself."""
    records = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [
        record["feature_name"]
        for record in records
        if record["availability_stage"] in {"POST_BUILD", "POST_FAILURE"}
        and record["feature_name"] != "tr_status"
    ]


def _append_post_build_features(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train-only encode post-build fields for the deliberately leaky regime."""
    post_columns = _post_build_columns()
    raw = pd.read_parquet(DATA_PATH, columns=post_columns)
    train = x_train.copy()
    test = x_test.copy()
    for column in post_columns:
        train_values = raw.loc[x_train.index, column]
        test_values = raw.loc[x_test.index, column]
        if pd.api.types.is_numeric_dtype(train_values):
            encoded_name = f"leaky_{column}"
            train[encoded_name] = pd.to_numeric(train_values, errors="coerce")
            test[encoded_name] = pd.to_numeric(test_values, errors="coerce")
        else:
            encoded_name = f"leaky_{column}_enc"
            encoder = LabelEncoder().fit(train_values.fillna("__missing__").astype(str))
            fallback = "__missing__" if "__missing__" in encoder.classes_ else encoder.classes_[0]
            safe_test = test_values.fillna("__missing__").astype(str)
            safe_test = safe_test.where(safe_test.isin(encoder.classes_), fallback)
            train[encoded_name] = encoder.transform(train_values.fillna("__missing__").astype(str))
            test[encoded_name] = encoder.transform(safe_test)
    return train, test, post_columns


def _fit_score(regime: str, x_train, y_train, x_test, y_test, note: str) -> dict:
    imputer = SimpleImputer(strategy="median")
    x_train_ready = imputer.fit_transform(x_train)
    x_test_ready = imputer.transform(x_test)
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
        "Regime": regime,
        "FeatureCount": int(x_train.shape[1]),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "AUC": float(roc_auc_score(y_test, probabilities)),
        "TrainingSeconds": float(elapsed),
        "Note": note,
    }


def _save_figure(results: pd.DataFrame) -> None:
    """Save a compact F1/AUC comparison for the evaluated availability regimes."""
    positions = range(len(results))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar([position - width / 2 for position in positions], results["F1"], width, label="F1", color="#2563eb")
    axis.bar([position + width / 2 for position in positions], results["AUC"], width, label="AUC", color="#dc2626")
    axis.set_xticks(list(positions), results["Regime"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Random Forest Performance by Feature Availability Regime")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def _save_analysis(results: pd.DataFrame, post_columns: list[str]) -> None:
    """Document the dataset-specific result without generalizing beyond it."""
    strict = results.loc[results["Regime"] == "Strict Pre-Build"].iloc[0]
    leaky = results.loc[results["Regime"] == "Post-Build / Leaky"].iloc[0]
    increase = float(leaky["F1"] - strict["F1"])
    relative = increase / float(strict["F1"]) * 100
    table = [
        "| Regime | Features | Precision | Recall | F1 | AUC | Training seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        table.append(
            f"| {row.Regime} | {row.FeatureCount} | {row.Precision:.4f} | {row.Recall:.4f} | "
            f"{row.F1:.4f} | {row.AUC:.4f} | {row.TrainingSeconds:.2f} |"
        )
    ANALYSIS_PATH.write_text(
        "# Leakage Regime Gradient Experiment\n\n"
        "All rows use the same 200,000-row cap, deterministic 80/20 stratified split, "
        "random seed, and Random Forest hyperparameters. The strict regime is exactly the production `FEATURE_COLS`.\n\n"
        + "\n".join(table)
        + "\n\n"
        + f"For this dataset, adding post-build log/duration information changes F1 from {strict.F1:.4f} "
        + f"to {leaky.F1:.4f}: an absolute increase of {increase:.4f} points and a relative increase of {relative:.1f}%. "
        + "This result is limited to this TravisTorrent export and this controlled feature/split configuration.\n\n"
        + "No distinct genuine during-build regime was evaluated: the registry identifies only `tr_jobs`, "
        + "a mutable, ambiguously timed collection rather than a timestamped in-flight measurement. It was not added.\n\n"
        + "Post-build/leaky fields: `"
        + "`, `".join(post_columns)
        + "`.\n",
        encoding="utf-8",
    )


def run_experiment() -> pd.DataFrame:
    """Run strict and deliberately leaky regimes, stopping on strict mismatch."""
    assert_features_available_for_prediction(FEATURE_COLS)
    preprocessing = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    if preprocessing["feature_cols"] != FEATURE_COLS:
        raise ValueError("Frozen preprocessing schema does not match FEATURE_COLS.")

    x_train, x_test, y_train, y_test = _recreate_split(DATA_PATH, FEATURE_COLS, encoders)
    strict = _fit_score(
        "Strict Pre-Build",
        x_train,
        y_train,
        x_test,
        y_test,
        "Exactly the 13 production pre-build features.",
    )
    if (
        abs(strict["F1"] - PRODUCTION_F1) > REPRODUCTION_TOLERANCE
        or abs(strict["AUC"] - PRODUCTION_AUC) > REPRODUCTION_TOLERANCE
    ):
        raise RuntimeError(
            "Strict pre-build regime did not reproduce frozen production metrics: "
            f"F1={strict['F1']:.6f} (expected {PRODUCTION_F1:.6f}), "
            f"AUC={strict['AUC']:.6f} (expected {PRODUCTION_AUC:.6f})."
        )

    leaky_train, leaky_test, post_columns = _append_post_build_features(x_train, x_test)
    leaky = _fit_score(
        "Post-Build / Leaky",
        leaky_train,
        y_train,
        leaky_test,
        y_test,
        "Strict features plus registry-labelled post-build/post-failure log and duration fields.",
    )
    results = pd.DataFrame([strict, leaky])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    _save_figure(results)
    _save_analysis(results, post_columns)
    print(results.to_string(index=False))
    print(f"Saved research-only outputs under {REPORTS_DIR}")
    return results


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
