"""Reproducible evaluation and reporting for saved CI/CD prediction models."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import joblib

# Some non-interactive environments have no writable user cache directory.
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "cicd-ai-optimizer-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = str(Path(tempfile.gettempdir()) / "cicd-ai-optimizer-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.features.engineer import engineer_features
from src.models.train import ARTIFACTS_DIR, DATA_PATH, MAX_TRAINING_ROWS, RANDOM_STATE


FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
MODEL_NAMES = ("random_forest", "decision_tree", "svm", "xgboost")
METRIC_COLUMNS = ("Accuracy", "Precision", "Recall", "F1", "AUC")
# Keep CV process-safe on restricted hosts; individual estimators retain their configured parallelism.
CV_N_JOBS = int(os.getenv("CV_N_JOBS", "1"))


def _load_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    """Load all fitted estimators and the training preprocessing bundle."""
    missing = [
        name
        for name in (*[f"{model}.joblib" for model in MODEL_NAMES], "preprocessing.joblib")
        if not (ARTIFACTS_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing training artifact(s): {', '.join(missing)}. "
            "Run `python -m src.models.train` first."
        )
    models = {name: joblib.load(ARTIFACTS_DIR / f"{name}.joblib") for name in MODEL_NAMES}
    return models, joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")


def _recreate_split(data_path: Path, feature_columns: list[str], encoders: dict):
    """Recreate training's exact capped 80/20 stratified split."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. Run the data pipeline first."
        )
    df = pd.read_parquet(data_path, engine="pyarrow")
    df, _ = engineer_features(df, encoders=encoders, fit_encoders=False, save_encoders=False)
    if len(df) > MAX_TRAINING_ROWS:
        df = df.sample(n=MAX_TRAINING_ROWS, random_state=RANDOM_STATE)

    target = df["tr_status"].eq("failed").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_columns],
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    return x_train, x_test, y_train, y_test


def _save_model_comparison(results: pd.DataFrame) -> None:
    """Save a grouped bar chart containing all primary test metrics."""
    axes = results.set_index("Model")[list(METRIC_COLUMNS)].plot(
        kind="bar", figsize=(12, 6), ylim=(0, 1), rot=0
    )
    axes.set_title("Model Performance Comparison")
    axes.set_ylabel("Score")
    axes.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close()


def _save_confusion_matrices(y_test, predictions: dict[str, object]) -> None:
    """Save one confusion-matrix panel for every trained model."""
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, name in zip(axes, MODEL_NAMES):
        matrix = confusion_matrix(y_test, predictions[name], labels=[0, 1])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
            xticklabels=["passed", "failed"],
            yticklabels=["passed", "failed"],
        )
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150)
    plt.close(figure)


def _save_feature_importance(model: object, feature_columns: list[str], filename: str, title: str) -> None:
    """Save the top 15 built-in feature importances for a tree model."""
    importance = pd.Series(model.feature_importances_, index=feature_columns).nlargest(15).sort_values()
    axis = importance.plot(kind="barh", figsize=(8, 6), color="#4c78a8")
    axis.set_title(title)
    axis.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close()


def _save_precision_recall_curves(y_test, probabilities: dict[str, object]) -> None:
    """Save overlaid precision-recall curves for the four models."""
    figure, axis = plt.subplots(figsize=(8, 6))
    for name in MODEL_NAMES:
        precision, recall, _ = precision_recall_curve(y_test, probabilities[name])
        axis.plot(recall, precision, label=name.replace("_", " ").title())
    axis.set_title("Precision-Recall Curves")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=150)
    plt.close(figure)


def evaluate_models(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Evaluate saved models, write report artifacts, and return result metrics."""
    models, preprocessing = _load_artifacts()
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    feature_columns = preprocessing["feature_cols"]
    x_train, x_test, y_train, y_test = _recreate_split(data_path, feature_columns, encoders)
    imputer = preprocessing["imputer"]
    x_test_imputed = imputer.transform(x_test)

    predictions: dict[str, object] = {}
    probabilities: dict[str, object] = {}
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        model_input = preprocessing["scaler"].transform(x_test_imputed) if name == "svm" else x_test_imputed
        prediction = model.predict(model_input)
        probability = model.predict_proba(model_input)[:, 1]
        predictions[name] = prediction
        probabilities[name] = probability
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Precision": precision_score(y_test, prediction, zero_division=0),
                "Recall": recall_score(y_test, prediction, zero_division=0),
                "F1": f1_score(y_test, prediction, zero_division=0),
                "AUC": roc_auc_score(y_test, probability),
            }
        )

    results = pd.DataFrame(rows, columns=["Model", *METRIC_COLUMNS])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ARTIFACTS_DIR / "model_results.csv", index=False)

    # SVM CV is intentionally omitted because RBF SVC is prohibitively slow at this scale.
    cv_scores: dict[str, dict[str, float]] = {}
    for name in ("random_forest", "decision_tree", "xgboost"):
        cv_pipeline = Pipeline([("imputer", clone(imputer)), ("model", clone(models[name]))])
        cv_scores[name] = {
            "f1": cross_val_score(cv_pipeline, x_train, y_train, cv=5, scoring="f1", n_jobs=CV_N_JOBS).mean(),
            "roc_auc": cross_val_score(cv_pipeline, x_train, y_train, cv=5, scoring="roc_auc", n_jobs=CV_N_JOBS).mean(),
        }
        print(f"{name} 5-fold CV: F1={cv_scores[name]['f1']:.4f}, AUC={cv_scores[name]['roc_auc']:.4f}")

    _save_model_comparison(results)
    _save_confusion_matrices(y_test, predictions)
    _save_feature_importance(models["random_forest"], feature_columns, "feature_importance_rf.png", "Random Forest Feature Importance")
    _save_feature_importance(models["xgboost"], feature_columns, "feature_importance_xgb.png", "XGBoost Feature Importance")
    _save_precision_recall_curves(y_test, probabilities)

    best_model = results.loc[results["F1"].idxmax(), "Model"]
    (ARTIFACTS_DIR / "best_model.txt").write_text(f"{best_model}\n", encoding="utf-8")
    print("\nModel results:\n" + results.to_string(index=False))
    print(f"\nBest model by F1: {best_model}")
    return results


def main() -> None:
    """Run the evaluation and reporting pipeline from the command line."""
    evaluate_models()


if __name__ == "__main__":
    main()
