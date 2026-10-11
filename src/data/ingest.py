"""Raw TravisTorrent data loading utilities."""

from pathlib import Path

import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """Load one compressed TravisTorrent export and report its size.

    The CSV is read exactly once. Callers should pass the returned DataFrame to
    preprocessing rather than reopening the input file.
    """
    input_path = Path(path)
    df = pd.read_csv(input_path, compression="gzip", low_memory=False)
    memory_mb = df.memory_usage(index=True, deep=True).sum() / (1024**2)
    print(f"Loaded raw data: shape={df.shape}, memory={memory_mb:.2f} MiB")
    return df
