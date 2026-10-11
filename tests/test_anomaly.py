"""Tests for isolation-forest anomaly detection."""

import numpy as np
import pandas as pd

from src.models.anomaly import AnomalyDetector


def test_obvious_outlier_is_flagged_and_detector_round_trips(tmp_path) -> None:
    normal_builds = np.zeros((100, 2))
    obvious_outlier = np.array([[100.0, 100.0]])
    features = pd.DataFrame(np.vstack([normal_builds, obvious_outlier]), columns=["f1", "f2"])

    detector = AnomalyDetector(contamination=0.01).fit(features)
    assert detector.predict(features)[-1] == -1
    assert detector.anomaly_scores(features)[-1] < detector.anomaly_scores(features)[0]

    artifact_path = tmp_path / "anomaly_detector.joblib"
    detector.save(str(artifact_path))
    assert AnomalyDetector.load(str(artifact_path)).predict(features)[-1] == -1
