"""Summarize saved walk-forward predictions at a per-repository alert budget.

This uses only existing parquet predictions.  Repository-cluster intervals
resample whole repositories, never individual builds and never refit models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT=Path(__file__).resolve().parents[1]
PARTS=ROOT / "reports" / "phase1_v2_parts"
OUT=ROOT / "reports" / "phase1_v2_walkforward_within_repo_metrics.csv"
SEED=20261013
BOOTSTRAPS=1000


def per_repo(pred: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for repo,d in pred.groupby("repo",sort=False):
        y=d.label.to_numpy(int); p=d.probability.to_numpy(float)
        if not y.sum(): continue
        # A within-repository constant score (as in repo_prior) has no
        # meaningful top-k ranking; stable sort order must not be mistaken
        # for alert-budget performance.
        if np.unique(p).size == 1:
            rows.append({"repo":repo,
                         "macro_within_repo_auc":0.5 if np.unique(y).size==2 else np.nan,
                         "macro_within_repo_recall_at_15":np.nan,
                         "macro_within_repo_precision_at_15":np.nan,
                         "macro_within_repo_lift_at_15":np.nan})
            continue
        k=max(1,int(np.ceil(.15*len(d)))); chosen=np.argsort(-p,kind="mergesort")[:k]
        rows.append({"repo":repo,
                     "macro_within_repo_auc":roc_auc_score(y,p) if np.unique(y).size==2 else np.nan,
                     "macro_within_repo_recall_at_15":y[chosen].sum()/y.sum(),
                     "macro_within_repo_precision_at_15":y[chosen].mean(),
                     "macro_within_repo_lift_at_15":y[chosen].mean()/y.mean()})
    return pd.DataFrame(rows)


def intervals(values: np.ndarray, rng: np.random.Generator) -> tuple[float,float]:
    values=values[np.isfinite(values)]
    if len(values)<2: return np.nan,np.nan
    draws=[rng.choice(values,size=len(values),replace=True).mean() for _ in range(BOOTSTRAPS)]
    return float(np.quantile(draws,.025)),float(np.quantile(draws,.975))


def main() -> None:
    rows=[]
    for fold in range(6,11):
        for tier,kind in (("A","rf"),("C","rf"),("Cplus","rf"),("C","repo_prior"),("C","historyonly"),
                          ("Dcommit","rf"),("Dcontext","rf"),("Dnorm","rf"),("DnormV2","rf"),("Dall","rf"),("DallV2","rf"),("C","balanced_rf")):
            path=PARTS/f"travis_walk_{fold}_{tier}_{kind}__finish_buffer_hours_0.parquet"
            if not path.exists(): continue
            metric=per_repo(pd.read_parquet(path)); rng=np.random.default_rng(SEED+fold)
            row={"fold":fold,"tier":tier,"model":kind,"repositories":len(metric),"finish_buffer_hours":0}
            for col in metric.columns[1:]:
                row[col]=float(metric[col].mean())
                low,high=intervals(metric[col].to_numpy(float),rng)
                row[f"{col}_ci_low"]=low; row[f"{col}_ci_high"]=high
            rows.append(row)
    result=pd.DataFrame(rows); result.to_csv(OUT,index=False); print(result.to_string(index=False))


if __name__=="__main__": main()
