"""Exploratory (not pre-registered): does a naive stream-persistence score explain Tier E's GHA gain?

    python scripts/analyze_gha_persistence.py <tag> <raw.csv> [<raw.csv> ...]

Reads the saved fold predictions in reports/phase1_v2_<tag>_parts/.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_gha_tier_e as ev
from compare_phase1_walkforward_tiers import per_repository_metrics

tag, raw = sys.argv[1], [Path(p) for p in sys.argv[2:]]
ev.configure(tag, raw, ev.PROTOCOL)
runs, _ = ev.load_runs("primary")
data = ev.with_history(runs, 0)
folds = ev.chronological_folds(data.gh_build_started_at)
parts = ROOT / "reports" / f"phase1_v2_{tag}_parts"
rows, per_repo = [], []
for fold in ev.FOLDS:
    train, test = data.iloc[np.concatenate(folds[:fold - 1])], data.iloc[folds[fold - 1]]
    prior = train.target_failed.mean()
    # Previous finished run on this stream, ties broken by the stream's last-5 rate.
    persist = test.hist_stream_prev_failed.fillna(prior) + 0.01 * test.hist_stream_fail_rate_5.fillna(prior)
    base = pd.DataFrame({"repo": test.gh_project_name.to_numpy(), "build_id": test.tr_build_id.to_numpy(),
                         "label": test.target_failed.to_numpy(), "probability": persist.to_numpy()})
    scores = {"persistence": base}
    for v in ("C_rf", "E_lgbmblend"):
        scores[v] = pd.read_parquet(parts / f"primary_0h_fold{fold}_{v}.parquet")
    m = {k: per_repository_metrics(s).set_index("repo").within_repo_auc for k, s in scores.items()}
    row = {"fold": fold, **{k: float(v.mean()) for k, v in m.items()},
           "stream_history_coverage": float(test.hist_stream_prev_failed.notna().mean())}
    rows.append(row)
    per_repo.append(pd.DataFrame(m).assign(fold=fold))
print(pd.DataFrame(rows).round(3).to_string(index=False))
pr = pd.concat(per_repo).reset_index().rename(columns={"index": "repo"})
pr["delta"] = pr.E_lgbmblend - pr.C_rf
fr = runs.groupby("gh_project_name").target_failed.mean().rename("fail_rate")
agg = pr.groupby("repo")[["C_rf", "E_lgbmblend", "persistence", "delta"]].mean().join(fr).sort_values("delta")
print(agg.round(3).to_string())
print("corr(delta, fail_rate) =", round(agg.delta.corr(agg.fail_rate), 3))
