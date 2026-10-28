"""Paired saved-prediction robustness comparison for Tier-C finish buffers."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from compare_phase1_walkforward_tiers import PARTS, SEED, bootstrap_delta, per_repository_metrics


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "reports" / "phase1_v2_walkforward_finish_buffer_sensitivity.csv"
METRICS=("within_repo_auc","within_repo_recall_at_15","within_repo_lift_at_15")


def main() -> None:
    rows=[]
    for buffer in (1,6):
        for fold in range(6,11):
            base_path=PARTS/f"travis_walk_{fold}_C_rf__finish_buffer_hours_0.parquet"
            candidate_path=PARTS/f"travis_walk_{fold}_C_rf__finish_buffer_hours_{buffer}.parquet"
            if not candidate_path.exists(): continue
            base=pd.read_parquet(base_path); candidate=pd.read_parquet(candidate_path)
            aligned=base[["repo","build_id","label"]].merge(candidate[["repo","build_id","label"]],on=["repo","build_id"],suffixes=("_0h",f"_{buffer}h"),validate="one_to_one")
            assert len(aligned)==len(base)==len(candidate) and (aligned.label_0h==aligned[f"label_{buffer}h"]).all(), "buffer changed test population or labels"
            left=per_repository_metrics(base).set_index("repo"); right=per_repository_metrics(candidate).set_index("repo")
            repos=left.index.intersection(right.index); row={"fold":fold,"finish_buffer_hours":buffer,"repositories":len(repos)}
            rng=np.random.default_rng(SEED+100*buffer+fold)
            for metric in METRICS:
                delta=(right.loc[repos,metric]-left.loc[repos,metric]).to_numpy(float)
                row[f"baseline_0h_{metric}"]=float(left.loc[repos,metric].mean())
                row[f"candidate_buffer_{metric}"]=float(right.loc[repos,metric].mean())
                row[f"delta_{metric}"]=float(np.nanmean(delta))
                low,high=bootstrap_delta(delta,rng); row[f"delta_{metric}_ci_low"]=low; row[f"delta_{metric}_ci_high"]=high
            rows.append(row)
    result=pd.DataFrame(rows); result.to_csv(OUT,index=False); print(result.to_string(index=False))


if __name__=="__main__": main()
