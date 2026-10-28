"""New-breakage analysis: score saved walk-forward predictions only on builds
whose stream had not already failed.

A model that mostly learns "this branch/PR is already red" looks strong on all
builds but cannot help find fresh breakages. This rescoring separates the two.
No models are fitted; saved Tier E (E_lgbmblend) and Tier C (C_rf) fold
predictions are joined to the finish-time-safe stream history.

Strata (stream = PR number for PR builds, else branch; as in Tier E):
  known_pass     last stream outcome *known at trigger time* was a pass
                 (hist_stream_prev_failed == 0 under the same finish buffer)
  prev_pass      the immediately preceding stream build (by start time) passed,
                 whether or not it had finished; defines the subset only, never
                 a feature
  known_fail     last known stream outcome was a failure
  no_history     no completed stream build before the trigger

Within each stratum, metrics are computed per repository (top-15% budget is
taken within the stratum) and macro-averaged; CIs are repository-cluster
bootstraps, deltas paired by repository, matching the Tier E comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_phase1_walkforward_tiers import PARTS, SEED, bootstrap_delta, per_repository_metrics  # noqa: E402
from src.models.phase1_rigor_evaluation import _stream_keys, _walk_data  # noqa: E402

OUT = ROOT / "reports" / "phase1_v2_new_breakage.csv"
DELTAS = ROOT / "reports" / "phase1_v2_new_breakage_vs_c.csv"
METRICS = ("within_repo_auc", "within_repo_recall_at_15", "within_repo_lift_at_15")
UNITS = {0: ("C_rf", "E_lgbmblend", "C_historyonly"), 1: ("C_rf", "E_lgbmblend"), 6: ("C_rf", "E_lgbmblend")}


def strata(buffer: int) -> pd.DataFrame:
    data, _ = _walk_data(buffer)
    d = data[["tr_build_id", "gh_project_name", "gh_build_started_at", "target_failed", "hist_stream_prev_failed"]].copy()
    d["stream"] = _stream_keys(data)
    d = d.sort_values(["gh_project_name", "gh_build_started_at", "tr_build_id"], kind="mergesort")
    d["prev_label"] = d.groupby(["gh_project_name", "stream"], sort=False).target_failed.shift()
    known = d.hist_stream_prev_failed
    d["stratum_known"] = np.select([known.eq(0), known.eq(1)], ["known_pass", "known_fail"], "no_history")
    d["prev_pass"] = d.prev_label.eq(0)
    return d.set_index("tr_build_id")[["target_failed", "stratum_known", "prev_pass"]]


def subsets(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"all": pred, "known_pass": pred[pred.stratum_known.eq("known_pass")],
            "prev_pass": pred[pred.prev_pass], "known_fail": pred[pred.stratum_known.eq("known_fail")],
            "no_history": pred[pred.stratum_known.eq("no_history")]}


def main() -> None:
    rows, deltas = [], []
    for buffer, units in UNITS.items():
        s = strata(buffer)
        for fold in range(6, 11):
            per = {}
            for unit in units:
                pred = pd.read_parquet(PARTS / f"travis_walk_{fold}_{unit}__finish_buffer_hours_{buffer}.parquet")
                pred = pred.join(s, on="build_id", how="left")
                if pred.stratum_known.isna().any() or not pred.label.eq(pred.target_failed).all():
                    raise AssertionError(f"{unit} fold {fold}: predictions do not match reconstructed builds")
                per[unit] = {name: per_repository_metrics(sub).set_index("repo") for name, sub in subsets(pred).items()}
                rng = np.random.default_rng(SEED + fold)
                for name, sub in subsets(pred).items():
                    m = per[unit][name]
                    row = {"fold": fold, "role": "development" if fold == 6 else "confirmation", "finish_buffer_hours": buffer,
                           "model": unit, "stratum": name, "builds": len(sub), "failures": int(sub.label.sum()),
                           "prevalence": float(sub.label.mean()) if len(sub) else np.nan,
                           "repositories_auc": int(m.within_repo_auc.notna().sum())}
                    for metric in METRICS:
                        v = m[metric].to_numpy(float)
                        row[metric] = float(np.nanmean(v))
                        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_delta(v, rng)
                    rows.append(row)
            rng = np.random.default_rng(SEED + 100 + fold)
            for name in per["C_rf"]:
                c, e = per["C_rf"][name], per["E_lgbmblend"][name]
                repos = c.index.intersection(e.index)
                row = {"fold": fold, "finish_buffer_hours": buffer, "stratum": name, "comparison": "E_lgbmblend_minus_C_rf",
                       "repositories": len(repos)}
                for metric in METRICS:
                    diff = (e.loc[repos, metric] - c.loc[repos, metric]).to_numpy(float)
                    row[f"delta_{metric}"] = float(np.nanmean(diff))
                    row[f"delta_{metric}_ci_low"], row[f"delta_{metric}_ci_high"] = bootstrap_delta(diff, rng)
                deltas.append(row)
            print(f"buffer {buffer}h fold {fold} done", flush=True)
    out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
    dl = pd.DataFrame(deltas); dl.to_csv(DELTAS, index=False)
    conf = out[out.role.eq("confirmation")]
    print(conf.groupby(["finish_buffer_hours", "stratum", "model"])[["prevalence", *METRICS]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
