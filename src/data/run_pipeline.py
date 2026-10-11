"""Command-line entry point for the data preparation pipeline."""

import argparse

from src.data.ingest import load_raw
from src.data.preprocess import STATUS_COLUMN, TIMESTAMP_COLUMN, clean, save_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw TravisTorrent build data.")
    parser.add_argument("--input", required=True, help="Path to the raw gzip CSV file.")
    parser.add_argument("--output", required=True, help="Destination parquet file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = clean(load_raw(args.input))
    save_processed(processed, args.output)

    total_rows = len(processed)
    failure_rate = (processed[STATUS_COLUMN].eq("failed").mean() * 100) if total_rows else 0.0
    date_range = (
        f"{processed[TIMESTAMP_COLUMN].min()} to {processed[TIMESTAMP_COLUMN].max()}"
        if total_rows
        else "no valid timestamps"
    )
    print(
        "Pipeline summary: "
        f"total rows={total_rows:,}, failure rate={failure_rate:.2f}%, date range={date_range}"
    )


if __name__ == "__main__":
    main()
