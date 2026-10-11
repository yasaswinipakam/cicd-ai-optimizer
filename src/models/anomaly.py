"""Reusable Isolation Forest wrapper for anomalous CI/CD build detection."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Detect unusual builds using sklearn's normal/anomaly label convention."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, x: pd.DataFrame) -> "AnomalyDetector":
        """Fit the detector and return this instance."""
        self.model.fit(x)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return 1 for normal builds and -1 for anomalies."""
        return self.model.predict(x)

    def anomaly_scores(self, x: pd.DataFrame) -> np.ndarray:
        """Return decision scores; lower values indicate more anomalous builds."""
        return self.model.decision_function(x)

    def save(self, path: str) -> None:
        """Persist this fitted detector with joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        """Load a detector persisted by :meth:`save`."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact at {path} is not an {cls.__name__}.")
        return loaded
