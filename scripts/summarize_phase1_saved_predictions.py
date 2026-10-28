"""Compute within-repository metrics from Phase-1 saved predictions only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.phase1_rigor_evaluation import PARTS, REPORTS


def main() -> None:
    rows=[]
    for path in sorted(PARTS.glob("*.json")):
        meta=json.loads(path.read_text(encoding="utf-8"))[0]
        pred=pd.read_parquet(path.with_suffix(".parquet"))
        values=[]
        for repo,d in pred.groupby("repo", dropna=False):
            if d.label.nunique() != 2: continue
            values.append({"repository":repo,"builds":len(d),"failures":int(d.label.sum()),"prevalence":float(d.label.mean()),"auc":float(roc_auc_score(d.label,d.probability))})
        if not values: continue
        detail=pd.DataFrame(values)
        detail.insert(0,"unit",meta["unit"]); detail.insert(1,"finish_buffer_hours",meta.get("finish_buffer_hours",0))
        detail.to_csv(PARTS / f"{path.stem}__within_repository_auc.csv",index=False)
        rows.append({"unit":meta["unit"],"finish_buffer_hours":meta.get("finish_buffer_hours",0),"dataset":meta.get("dataset"),"split":meta.get("split"),"tier":meta.get("tier"),"model":meta.get("model"),"repositories_with_both_classes":len(detail),"macro_within_repo_auc":float(detail.auc.mean()),"macro_within_repo_auc_sd":float(detail.auc.std(ddof=1)) if len(detail)>1 else np.nan,"pooled_auc":meta.get("roc_auc")})
    pd.DataFrame(rows).to_csv(REPORTS / "phase1_v2_within_repository_auc.csv",index=False)
    print(f"Wrote within-repository summaries for {len(rows)} saved parts")


if __name__ == "__main__": main()
