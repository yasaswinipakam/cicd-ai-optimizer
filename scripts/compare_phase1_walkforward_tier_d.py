"""Paired repository-bootstrap comparison of frozen Tier-D variants vs Tier C."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from compare_phase1_walkforward_tiers import PARTS, SEED, bootstrap_delta, per_repository_metrics


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "reports" / "phase1_v2_walkforward_tier_d_vs_c.csv"
VARIANTS=("Dcommit","Dcontext","DnormV2","DallV2")
METRICS=("within_repo_auc","within_repo_recall_at_15","within_repo_lift_at_15")


def main() -> None:
    rows=[]
    for fold in range(6,11):
        baseline=per_repository_metrics(pd.read_parquet(PARTS/f"travis_walk_{fold}_C_rf__finish_buffer_hours_0.parquet")).set_index("repo")
        for tier in VARIANTS:
            candidate=per_repository_metrics(pd.read_parquet(PARTS/f"travis_walk_{fold}_{tier}_rf__finish_buffer_hours_0.parquet")).set_index("repo")
            repos=baseline.index.intersection(candidate.index)
            row={"fold":fold,"comparison":f"{tier}_minus_C","repositories":len(repos)}
            rng=np.random.default_rng(SEED+fold)
            for metric in METRICS:
                difference=(candidate.loc[repos,metric]-baseline.loc[repos,metric]).to_numpy(float)
                row[f"delta_{metric}"]=float(np.nanmean(difference))
                low,high=bootstrap_delta(difference,rng)
                row[f"delta_{metric}_ci_low"]=low; row[f"delta_{metric}_ci_high"]=high
            rows.append(row)
    result=pd.DataFrame(rows); result.to_csv(OUT,index=False); print(result.to_string(index=False))


if __name__=="__main__": main()
