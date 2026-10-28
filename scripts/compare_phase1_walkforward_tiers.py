"""Paired, repository-cluster comparison of saved Tier C and Tier Cplus scores.

No models are fitted here.  A repository's alert budget is its own top 15%
of builds, so this reports the requested macro within-repository metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "reports" / "phase1_v2_parts"
OUT = ROOT / "reports" / "phase1_v2_walkforward_cplus_vs_c.csv"
SEED = 20261013
BOOTSTRAPS = 1000


def per_repository_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repo, d in predictions.groupby("repo", sort=False):
        y = d.label.to_numpy(int)
        p = d.probability.to_numpy(float)
        if len(d) == 0 or y.sum() == 0:
            continue
        if np.unique(p).size == 1:
            rows.append({"repo": repo,
                         "within_repo_recall_at_15": np.nan,
                         "within_repo_precision_at_15": np.nan,
                         "within_repo_lift_at_15": np.nan,
                         "within_repo_auc": 0.5 if np.unique(y).size == 2 else np.nan})
            continue
        k = max(1, int(np.ceil(.15 * len(d))))
        selected = np.argsort(-p, kind="mergesort")[:k]
        rows.append({
            "repo": repo,
            "within_repo_recall_at_15": y[selected].sum() / y.sum(),
            "within_repo_precision_at_15": y[selected].mean(),
            "within_repo_lift_at_15": y[selected].mean() / y.mean(),
            "within_repo_auc": roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan,
        })
    return pd.DataFrame(rows)


def bootstrap_delta(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    samples = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(BOOTSTRAPS)]
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def main() -> None:
    records = []
    for fold in range(6, 11):
        c = pd.read_parquet(PARTS / f"travis_walk_{fold}_C_rf__finish_buffer_hours_0.parquet")
        plus = pd.read_parquet(PARTS / f"travis_walk_{fold}_Cplus_rf__finish_buffer_hours_0.parquet")
        aligned = c[["repo", "build_id", "label"]].merge(
            plus[["repo", "build_id", "label"]], on=["repo", "build_id"], suffixes=("_c", "_plus"), validate="one_to_one"
        )
        assert len(aligned) == len(c) == len(plus) and (aligned.label_c == aligned.label_plus).all(), "saved predictions do not describe the same test builds"
        left = per_repository_metrics(c).set_index("repo")
        right = per_repository_metrics(plus).set_index("repo")
        common = left.index.intersection(right.index)
        rng = np.random.default_rng(SEED + fold)
        row = {"fold": fold, "repositories": len(common)}
        for metric in ("within_repo_auc", "within_repo_recall_at_15", "within_repo_precision_at_15", "within_repo_lift_at_15"):
            delta = (right.loc[common, metric] - left.loc[common, metric]).to_numpy(float)
            row[f"c_macro_{metric}"] = float(left.loc[common, metric].mean())
            row[f"cplus_macro_{metric}"] = float(right.loc[common, metric].mean())
            row[f"cplus_minus_c_macro_{metric}"] = float(np.nanmean(delta))
            low, high = bootstrap_delta(delta, rng)
            row[f"cplus_minus_c_macro_{metric}_ci_low"] = low
            row[f"cplus_minus_c_macro_{metric}_ci_high"] = high
        records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
