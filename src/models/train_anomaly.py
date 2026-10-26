"""Fit the anomaly-detection layer on the reproducible held-out build split."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import joblib
import matplotlib

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "cicd-ai-optimizer-matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = str(Path(tempfile.gettempdir()) / "cicd-ai-optimizer-cache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.models.anomaly import AnomalyDetector
from src.models.evaluate import _recreate_split
from src.models.train import ARTIFACTS_DIR, DATA_PATH


FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
ANOMALY_PATH = ARTIFACTS_DIR / "anomaly_detector.joblib"


def _failure_rate(target: pd.Series, mask) -> float:
    """Return the failed-build fraction for one prediction group."""
    subset = target.loc[mask]
    return float(subset.mean()) if len(subset) else 0.0


def _save_chart(anomaly_count: int, normal_count: int, anomaly_rate: float, normal_rate: float) -> None:
    """Save anomaly counts and failure-rate comparison as a two-panel chart."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Anomalous", "Normal"]
    colors = ["#e45756", "#4c78a8"]
    axes[0].bar(labels, [anomaly_count, normal_count], color=colors)
    axes[0].set_title("Build Anomaly Counts")
    axes[0].set_ylabel("Builds")
    axes[1].bar(labels, [anomaly_rate * 100, normal_rate * 100], color=colors)
    axes[1].set_title("Build Failure Rate by Group")
    axes[1].set_ylabel("Failure rate (%)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "anomaly_detection.png", dpi=150)
    plt.close(figure)


def train_anomaly_detector(data_path: Path = DATA_PATH) -> AnomalyDetector:
    """Fit and save the anomaly detector using Task 4's held-out test split."""
    preprocessing_path = ARTIFACTS_DIR / "preprocessing.joblib"
    if not preprocessing_path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifact not found: {preprocessing_path}. Run model training first."
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run the data pipeline first.")

    preprocessing = joblib.load(preprocessing_path)
    encoders = joblib.load(ARTIFACTS_DIR / "encoders.joblib")
    # Same history, cold-start prior and held-out split as model evaluation.
    _x_train, x_test, _y_train, y_test = _recreate_split(
        data_path, preprocessing["feature_cols"], encoders, preprocessing["cold_start_prior"]
    )

    x_test_imputed = preprocessing["imputer"].transform(x_test)
    detector = AnomalyDetector().fit(x_test_imputed)
    predictions = detector.predict(x_test_imputed)
    anomalous = predictions == -1
    normal = predictions == 1
    anomaly_count, normal_count = int(anomalous.sum()), int(normal.sum())
    anomaly_failure_rate = _failure_rate(y_test, anomalous)
    normal_failure_rate = _failure_rate(y_test, normal)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    detector.save(str(ANOMALY_PATH))
    _save_chart(anomaly_count, normal_count, anomaly_failure_rate, normal_failure_rate)
    print(f"Anomalies detected: {anomaly_count:,} of {len(predictions):,}")
    print(f"Failure rate among anomalous builds: {anomaly_failure_rate:.2%}")
    print(f"Failure rate among normal builds: {normal_failure_rate:.2%}")
    if anomaly_failure_rate <= normal_failure_rate:
        print("WARNING: anomaly failure rate is not higher than the normal-build rate; discuss this limitation.")
    return detector


def main() -> None:
    """Run anomaly-detector training from the command line."""
    train_anomaly_detector()


if __name__ == "__main__":
    main()
