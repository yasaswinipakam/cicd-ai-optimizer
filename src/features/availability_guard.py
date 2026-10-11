"""Runtime enforcement of the feature-availability registry."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable


REGISTRY_PATH = Path(__file__).resolve().parents[2] / "reports" / "feature_availability_registry.json"


class FeatureAvailabilityError(ValueError):
    """Raised when a feature is unknown or unavailable at prediction time."""


@lru_cache(maxsize=1)
def load_feature_availability_registry() -> dict[str, dict]:
    """Load records keyed by feature name from the versioned registry artifact."""
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return {record["feature_name"]: record for record in records}


def assert_features_available_for_prediction(columns: Iterable[str]) -> None:
    """Reject unknown, post-build, or otherwise unsafe model-input columns."""
    registry = load_feature_availability_registry()
    unsafe: list[str] = []
    for column in columns:
        record = registry.get(column)
        if record is None:
            unsafe.append(f"{column} (not registered)")
        elif not record["available_at_prediction_time"]:
            unsafe.append(
                f"{column} ({record['availability_stage']}; {record['reason']})"
            )
    if unsafe:
        raise FeatureAvailabilityError(
            "Unsafe prediction feature(s) blocked by the availability registry: "
            + "; ".join(unsafe)
        )
