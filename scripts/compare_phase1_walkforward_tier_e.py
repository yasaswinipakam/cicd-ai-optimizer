"""Paired repository-bootstrap comparison of frozen Tier-E variants vs Tier C RF.

Fold 6 was the development fold for Tier E; folds 7-10 are confirmation.
No models are fitted here; only saved walk-forward predictions are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_phase1_walkforward_tiers import PARTS, SEED, bootstrap_delta, per_repository_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase1_v2_walkforward_tier_e_vs_c.csv"
SUMMARY = ROOT / "reports" / "phase1_v2_walkforward_tier_e_summary.csv"
VARIANTS = ("C_lgbm", "C_lgbmrank", "E_lgbm", "E_lgbmrank", "E_lgbmblend", "Enojobs_lgbmblend")
METRICS = ("within_repo_auc", "within_repo_recall_at_15", "within_repo_lift_at_15")


def _part(unit: str, buffer: int) -> Path:
    return PARTS / f"{unit}__finish_buffer_hours_{buffer}"


def main() -> None:
    rows, summary = [], []
    comparisons = [(v, 0) for v in VARIANTS] + [("E_lgbmblend", 1), ("E_lgbmblend", 6)]
    for fold in range(6, 11):
        for variant, buffer in comparisons:
            path = _part(f"travis_walk_{fold}_{variant}", buffer)
            base_path = _part(f"travis_walk_{fold}_C_rf", buffer)
            if not path.with_suffix(".parquet").exists() or not base_path.with_suffix(".parquet").exists():
                continue
            cand_pred = pd.read_parquet(path.with_suffix(".parquet"))
            baseline = per_repository_metrics(pd.read_parquet(base_path.with_suffix(".parquet"))).set_index("repo")
            candidate = per_repository_metrics(cand_pred).set_index("repo")
            repos = baseline.index.intersection(candidate.index)
            row = {"fold": fold, "role": "development" if fold == 6 else "confirmation", "finish_buffer_hours": buffer,
                   "comparison": f"{variant}_minus_C_rf", "repositories": len(repos)}
            rng = np.random.default_rng(SEED + fold)
            for metric in METRICS:
                difference = (candidate.loc[repos, metric] - baseline.loc[repos, metric]).to_numpy(float)
                row[f"c_rf_{metric}"] = float(np.nanmean(baseline.loc[repos, metric]))
                row[f"candidate_{metric}"] = float(np.nanmean(candidate.loc[repos, metric]))
                row[f"delta_{metric}"] = float(np.nanmean(difference))
                row[f"delta_{metric}_ci_low"], row[f"delta_{metric}_ci_high"] = bootstrap_delta(difference, rng)
            rows.append(row)
            saved = json.loads(path.with_suffix(".json").read_text())[0]
            summary.append({"fold": fold, "variant": variant, "finish_buffer_hours": buffer,
                            "macro_within_repo_auc": saved["macro_within_repo_auc"], "pooled_auc": roc_auc_score(cand_pred.label, cand_pred.probability),
                            "pr_auc": saved["pr_auc"], "f1": saved["f1"], "brier": saved["brier"], "recall_at_15": saved["recall_at_15"]})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    s = pd.DataFrame(summary); s.to_csv(SUMMARY, index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
