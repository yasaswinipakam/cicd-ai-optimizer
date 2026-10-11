"""Unit tests for serializable real-time probability calibration."""

import numpy as np

from src.inference.calibration import ProbabilityCalibrator
from src.models.chronological_calibration import expected_calibration_error


def test_probability_calibrators_return_bounded_plain_numeric_scores() -> None:
    probabilities = np.array([0.05, 0.10, 0.30, 0.65, 0.85, 0.95])
    outcomes = np.array([0, 0, 0, 1, 1, 1])
    for method in ("raw", "sigmoid", "isotonic"):
        transformed = ProbabilityCalibrator.fit(method, probabilities, outcomes).transform(probabilities)
        assert transformed.shape == probabilities.shape
        assert np.all((transformed >= 0) & (transformed <= 1))
        assert all(isinstance(float(value), float) for value in transformed)


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_binary_scores() -> None:
    outcomes = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(outcomes, probabilities) == 0.0
