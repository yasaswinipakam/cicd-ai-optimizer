"""Small, serializable probability-calibration layer for binary CI risk scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibrator:
    """Map raw model scores to calibrated failure probabilities.

    ``sigmoid`` is Platt scaling and ``isotonic`` is a non-parametric monotonic
    mapping.  ``raw`` is an explicit identity mapping so every deployment has a
    clearly named probability treatment.
    """

    method: str = "raw"
    estimator: object | None = None

    @classmethod
    def fit(cls, method: str, raw_probabilities, targets) -> "ProbabilityCalibrator":
        raw = np.asarray(raw_probabilities, dtype=float).reshape(-1)
        y = np.asarray(targets, dtype=int).reshape(-1)
        if method == "raw":
            return cls(method="raw")
        if method == "sigmoid":
            estimator = LogisticRegression(random_state=42, solver="lbfgs")
            estimator.fit(raw.reshape(-1, 1), y)
            return cls(method=method, estimator=estimator)
        if method == "isotonic":
            estimator = IsotonicRegression(out_of_bounds="clip")
            estimator.fit(raw, y)
            return cls(method=method, estimator=estimator)
        raise ValueError("method must be one of: raw, sigmoid, isotonic")

    def transform(self, raw_probabilities) -> np.ndarray:
        raw = np.asarray(raw_probabilities, dtype=float).reshape(-1)
        if self.method == "raw":
            calibrated = raw
        elif self.method == "sigmoid":
            calibrated = self.estimator.predict_proba(raw.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            calibrated = self.estimator.predict(raw)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
