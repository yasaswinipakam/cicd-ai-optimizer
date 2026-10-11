"""Cleaning and persistence routines for CI/CD build data."""

from pathlib import Path

import pandas as pd


TIMESTAMP_COLUMN = "gh_build_started_at"
STATUS_COLUMN = "tr_status"
VALID_STATUSES = {"passed", "failed"}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return build records suitable for failure-prediction analysis."""
    missing_columns = {STATUS_COLUMN, TIMESTAMP_COLUMN}.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing_columns))}")

    print(f"Starting clean: {len(df):,} rows")
    cleaned = df.loc[df[STATUS_COLUMN].isin(VALID_STATUSES)].copy()
    print(f"After status filter: {len(cleaned):,} rows")

    cleaned[TIMESTAMP_COLUMN] = pd.to_datetime(
        cleaned[TIMESTAMP_COLUMN], errors="coerce", utc=True
    )
    unparsable_count = int(cleaned[TIMESTAMP_COLUMN].isna().sum())
    print(f"After timestamp parse: {len(cleaned):,} rows ({unparsable_count:,} unparseable)")

    timestamp = cleaned[TIMESTAMP_COLUMN]
    cleaned["hour"] = timestamp.dt.hour
    cleaned["day_of_week"] = timestamp.dt.dayofweek
    cleaned["month"] = timestamp.dt.month
    cleaned["is_weekend"] = timestamp.dt.dayofweek.isin([5, 6])
    cleaned["is_night"] = timestamp.dt.hour.isin(list(range(0, 6)))
    print(f"After time feature derivation: {len(cleaned):,} rows")

    cleaned = cleaned.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    print(f"After invalid timestamp drop: {len(cleaned):,} rows")
    return cleaned


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save processed data as a Parquet dataset, creating parent folders."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved processed data to {output_path} ({len(df):,} rows)")
