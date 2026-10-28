"""Paired repository-cluster differences for Tier C against saved baselines."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from compare_phase1_walkforward_tiers import per_repository_metrics, bootstrap_delta, PARTS, SEED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT / "reports" / "phase1_v2_walkforward_tier_c_vs_baselines.csv"
BASELINES=(("tier_a_rf", "A", "rf"), ("history_only_last5", "C", "historyonly"), ("repo_prior", "C", "repo_prior"))
METRICS=("within_repo_auc", "within_repo_recall_at_15", "within_repo_lift_at_15")


def main() -> None:
    rows=[]
    for fold in range(6,11):
        c=per_repository_metrics(pd.read_parquet(PARTS/f"travis_walk_{fold}_C_rf__finish_buffer_hours_0.parquet")).set_index("repo")
        for name,tier,kind in BASELINES:
            b=per_repository_metrics(pd.read_parquet(PARTS/f"travis_walk_{fold}_{tier}_{kind}__finish_buffer_hours_0.parquet")).set_index("repo")
            repos=c.index.intersection(b.index); row={"fold":fold,"comparison":f"tier_c_rf_minus_{name}","repositories":len(repos)}
            rng=np.random.default_rng(SEED+fold)
            for metric in METRICS:
                delta=(c.loc[repos,metric]-b.loc[repos,metric]).to_numpy(float)
                finite=delta[np.isfinite(delta)]
                row[f"delta_{metric}"]=float(finite.mean()) if len(finite) else np.nan
                low,high=bootstrap_delta(delta,rng); row[f"delta_{metric}_ci_low"]=low; row[f"delta_{metric}_ci_high"]=high
            rows.append(row)
    result=pd.DataFrame(rows); result.to_csv(OUT,index=False); print(result.to_string(index=False))


if __name__=="__main__": main()
