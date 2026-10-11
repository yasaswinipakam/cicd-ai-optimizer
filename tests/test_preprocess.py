"""Tests for the CI/CD data cleaning pipeline."""

import pandas as pd

from src.data.preprocess import clean


def test_clean_filters_statuses_creates_time_features_and_drops_bad_dates() -> None:
    source = pd.DataFrame(
        {
            "tr_status": [
                "passed",
                "failed",
                "errored",
                "canceled",
                "passed",
                "failed",
                "passed",
                "failed",
                "passed",
                "failed",
                "passed",
                "failed",
            ],
            "gh_build_started_at": [
                "2024-01-01T02:30:00Z",  # Monday, night
                "2024-01-06T18:00:00Z",  # Saturday
                "2024-01-02T10:00:00Z",
                "2024-01-03T10:00:00Z",
                "not-a-date",
                "2024-02-04T23:15:00Z",
                "2024-02-05T06:00:00Z",
                "2024-02-06T07:00:00Z",
                "2024-02-07T08:00:00Z",
                "2024-02-08T09:00:00Z",
                "2024-02-09T10:00:00Z",
                "2024-02-10T11:00:00Z",
            ],
        }
    )

    result = clean(source)

    assert set(result["tr_status"]) == {"passed", "failed"}
    assert len(result) == 9  # 10 valid statuses minus one unparsable timestamp
    assert {"hour", "day_of_week", "month", "is_weekend", "is_night"}.issubset(result.columns)

    monday_night = result.loc[result["gh_build_started_at"] == pd.Timestamp("2024-01-01T02:30:00Z")].iloc[0]
    assert monday_night["hour"] == 2
    assert monday_night["day_of_week"] == 0
    assert monday_night["month"] == 1
    assert bool(monday_night["is_weekend"]) is False
    assert bool(monday_night["is_night"]) is True

    saturday = result.loc[result["gh_build_started_at"] == pd.Timestamp("2024-01-06T18:00:00Z")].iloc[0]
    assert saturday["day_of_week"] == 5
    assert bool(saturday["is_weekend"]) is True
