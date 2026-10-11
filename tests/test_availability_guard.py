"""Regression tests for availability-aware prediction feature enforcement."""

import pytest

from src.features.availability_guard import (
    FeatureAvailabilityError,
    assert_features_available_for_prediction,
    load_feature_availability_registry,
)
from src.features.schema import FEATURE_COLS


def test_current_feature_schema_is_safe_for_prediction() -> None:
    assert_features_available_for_prediction(FEATURE_COLS)


def test_post_build_log_feature_is_rejected() -> None:
    with pytest.raises(FeatureAvailabilityError, match="tr_log_buildduration"):
        assert_features_available_for_prediction(["tr_log_buildduration"])


def test_registry_marks_all_current_model_columns_as_available() -> None:
    registry = load_feature_availability_registry()
    assert all(registry[column]["available_at_prediction_time"] for column in FEATURE_COLS)
    assert all(registry[column]["used_by_model"] for column in FEATURE_COLS)
