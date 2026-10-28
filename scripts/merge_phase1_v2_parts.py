"""Merge completed Phase-1 checkpoint parts without refitting a model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models.phase1_rigor_evaluation import BOOTSTRAPS, PARTS, REPORTS, SEED, cluster_ci


def main() -> None:
    rows=[]
    for path in sorted(PARTS.glob("*.json")):
        row=json.loads(path.read_text(encoding="utf-8"))[0]
        prediction_path=path.with_suffix(".parquet")
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing predictions for {path.stem}: {prediction_path}")
        predictions=pd.read_parquet(prediction_path)
        # CI is intentionally recomputed from saved predictions: no fitted model
        # is loaded or refit here.
        row.update(cluster_ci(predictions, row))
        rows.append(row)
    if not rows: raise SystemExit(f"No part files in {PARTS}")
    result=pd.DataFrame(rows)
    result["buffer_role"] = np.where(
        (result.get("tier") == "C") & (result.get("finish_buffer_hours", 0) == 0),
        "headline_p95_fallback_0h",
        np.where((result.get("tier") == "C"), "finish_buffer_sensitivity", "not_applicable"),
    )
    result=result.sort_values(["dataset","split","tier","model","unit","finish_buffer_hours"])
    result.to_csv(REPORTS / "phase1_v2_results.csv",index=False)
    print(f"Merged {len(result)} units; bootstrap repetitions={BOOTSTRAPS}, seed={SEED}")


if __name__ == "__main__": main()
