"""External test of the frozen Tier E recipe on GitHub Actions runs.

Implements reports/phase1_v2_gha_tier_e_protocol.md exactly; that file's
SHA-256 is recorded in the manifest.  GHA columns are renamed to the Travis
names so the unmodified Travis history, model and metric code is reused.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare_phase1_walkforward_tiers import SEED, bootstrap_delta, per_repository_metrics  # noqa: E402
from src.models.phase1_rigor_evaluation import (  # noqa: E402
    GHA_RAW, HISTORY, INFLIGHT, NORM_SOURCE, STREAM, _assert_history, _assert_split, _assert_stream_history,
    _stream_keys, add_safe_history, add_stream_history, fit_model, fit_predict_lgbm, repo_prior_scores,
)

REPORTS = ROOT / "reports"
PROTOCOL = REPORTS / "phase1_v2_gha_tier_e_protocol.md"
PARTS = REPORTS / "phase1_v2_gha_tier_e_parts"
RESULTS = REPORTS / "phase1_v2_gha_tier_e_results.csv"
DELTAS = REPORTS / "phase1_v2_gha_tier_e_vs_c.csv"
MANIFEST = REPORTS / "phase1_v2_gha_tier_e_manifest.json"
MAX_RUN_DAYS = 35
FOLDS = range(6, 11)
BASE_NUM = ["additions", "deletions", "total_churn", "files_modified", "msg_len", "is_merge", "num_parents",
            "time_since_last_commit", "hour", "day_of_week", "month", "is_weekend", "is_night", "is_pr",
            "is_default_branch"]
BASE_CAT = ["workflow"]
FEATURES = {
    "A": (BASE_NUM, BASE_CAT),
    "C": ([*BASE_NUM, *HISTORY], BASE_CAT),
    "E": ([*BASE_NUM, *HISTORY, *STREAM, *INFLIGHT], BASE_CAT),
}
PRIMARY = ("A_rf", "C_rf", "C_historyonly", "C_repoprior", "C_lgbmblend", "E_lgbm", "E_lgbmblend")
SECONDARY = ("C_rf", "E_lgbmblend")
# (label policy, finish buffer hours) -> variants scored
SETTINGS = {("primary", 0): PRIMARY, ("primary", 6): SECONDARY, ("valid_only", 0): SECONDARY}
METRICS = ("within_repo_auc", "within_repo_recall_at_15", "within_repo_lift_at_15")


def load_runs(label_policy: str) -> tuple[pd.DataFrame, dict]:
    columns = ["run_id", "repo", "workflow_id", "created_at", "updated_at", "conclusion", "run_attempt", "event",
               "head_branch", "actor_login", "head_sha", "duration", "additions", "deletions", "total_churn",
               "files_modified", "msg_len", "is_merge", "num_parents", "time_since_last_commit"]
    raw = pd.read_csv(GHA_RAW, usecols=columns, low_memory=False)
    for c in ("created_at", "updated_at"):
        raw[c] = pd.to_datetime(raw[c], utc=True, errors="coerce")
    d = raw.loc[raw.run_attempt.eq(1) & raw.event.isin(["push", "pull_request"])
                & raw.conclusion.isin(["success", "failure", "startup_failure"])
                & raw.workflow_id.notna() & raw.created_at.notna() & raw.updated_at.notna()].copy()
    counts = {"first_attempt_push_pr_runs": len(d)}
    too_long = (d.updated_at - d.created_at) > pd.Timedelta(days=MAX_RUN_DAYS)
    counts["dropped_completion_unknown"] = int(too_long.sum())
    d = d.loc[~too_long]
    if label_policy == "valid_only":
        never_ran = d.conclusion.eq("startup_failure") | (d.conclusion.eq("failure") & d.duration.le(0))
        counts["dropped_never_executed_proxy"] = int(never_ran.sum())
        d = d.loc[~never_ran]
    elif label_policy != "primary":
        raise ValueError(label_policy)
    d = d.sort_values(["created_at", "run_id"], kind="mergesort").reset_index(drop=True)
    is_pr = d.event.eq("pull_request")
    branch = d.head_branch.fillna("").astype(str)
    out = pd.DataFrame({
        "tr_build_id": d.run_id.astype(np.int64), "gh_project_name": d.repo, "gh_build_started_at": d.created_at,
        "completed_at": d.updated_at, "target_failed": d.conclusion.ne("success").astype(int),
        "git_branch": branch, "gh_is_pr": is_pr,
        # No PR number in GHA data: the PR author + head branch stands in for it.
        "gh_pull_req_num": np.where(is_pr, d.actor_login.fillna("").astype(str) + ":" + branch, None),
        "git_trigger_commit": d.head_sha, "workflow": d.workflow_id.astype(np.int64).astype(str), "job_count": 1.0,
        "is_pr": is_pr.astype(int), "is_default_branch": branch.isin(["main", "master"]).astype(int),
        "hour": d.created_at.dt.hour, "day_of_week": d.created_at.dt.dayofweek, "month": d.created_at.dt.month,
    })
    out["is_weekend"] = out.day_of_week.isin([5, 6]).astype(int)
    out["is_night"] = out.hour.between(0, 5).astype(int)
    for c in ("additions", "deletions", "total_churn", "files_modified", "msg_len", "num_parents", "time_since_last_commit"):
        out[c] = pd.to_numeric(d[c], errors="coerce").to_numpy()
    out["is_merge"] = d.is_merge.astype(float).to_numpy()
    out[NORM_SOURCE] = 0.0
    out["stream"] = _stream_keys(out)
    # tr_prev_build analogue: previous run of the same workflow on the same stream.
    group = out.groupby(["gh_project_name", "workflow", "stream"], sort=False)
    out["tr_prev_build"] = group.tr_build_id.shift()
    # New-breakage stratum: that previous run passed (defines a subset only, never a feature).
    out["prev_label"] = group.target_failed.shift()
    counts.update({"runs": len(out), "repositories": int(out.gh_project_name.nunique()),
                   "prevalence": float(out.target_failed.mean())})
    return out, counts


def _assert_stream_sample(d: pd.DataFrame, rows: int = 600) -> None:
    """Brute-force check on rows sampled from every repository, large ones included."""
    start = d.gh_build_started_at.dt.as_unit("ns").astype("int64").to_numpy()
    finish = d._finished_at.dt.as_unit("ns").astype("int64").to_numpy()
    ids = d.tr_build_id.to_numpy(np.int64); y = d.target_failed.to_numpy()
    repo = d.gh_project_name.to_numpy(); stream = d.stream.to_numpy()
    for i in np.random.default_rng(SEED).choice(len(d), size=min(rows, len(d)), replace=False):
        same_repo = repo == repo[i]
        earlier = same_repo & ((start < start[i]) | ((start == start[i]) & (ids < ids[i])))
        assert d.ctx_repo_pending.iat[i] == (earlier & (finish >= start[i])).sum(), f"in-flight count at row {i}"
        mask = same_repo & (stream == stream[i]) & (finish < start[i])
        prior = y[mask][np.lexsort((ids[mask], start[mask], finish[mask]))]
        if len(prior):
            assert d.hist_stream_prev_failed.iat[i] == prior[-1], f"stream outcome at row {i}"
            assert np.isclose(d.hist_stream_fail_rate_5.iat[i], prior[-5:].mean()), f"stream rate at row {i}"
        else:
            assert pd.isna(d.hist_stream_prev_failed.iat[i]), f"stream outcome before completion at row {i}"


def with_history(runs: pd.DataFrame, buffer: int) -> pd.DataFrame:
    d = runs.copy()
    d["_finished_at"] = d.completed_at + pd.Timedelta(hours=buffer)
    d = add_stream_history(add_safe_history(d))
    _assert_history(d); _assert_stream_history(d); _assert_stream_sample(d)
    return d


def score(variant: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    tier, kind = variant.split("_", 1)
    numeric, categorical = FEATURES[tier]
    if kind == "historyonly":
        return test.hist_repo_fail_rate_5.fillna(float(train.target_failed.mean())).to_numpy(float)
    if kind == "repoprior":
        return repo_prior_scores(train, test)
    if kind in ("lgbm", "lgbmblend"):
        return fit_predict_lgbm(kind, train, test, numeric, categorical)
    return fit_model(kind, train, numeric, categorical).predict_proba(test[numeric + categorical])[:, 1]


def chronological_folds(times: pd.Series, k: int = 10) -> list[np.ndarray]:
    """Ten equal folds, each boundary moved forward past runs sharing a creation time.

    GHA creation times have one-second resolution, so a plain equal split can
    put simultaneous runs on both sides, breaking strict train-before-test.
    """
    t = times.to_numpy(); bounds = [0]
    for b in (len(chunk) for chunk in np.array_split(np.arange(len(t)), k)):
        end = bounds[-1] + b
        while 0 < end < len(t) and t[end] == t[end - 1]:
            end += 1
        bounds.append(min(end, len(t)))
    bounds[-1] = len(t)
    return [np.arange(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]


def run_setting(label_policy: str, buffer: int, variants: tuple[str, ...], runs: pd.DataFrame) -> list[pd.DataFrame]:
    data = with_history(runs, buffer)
    folds = chronological_folds(data.gh_build_started_at)
    preds = []
    for fold in FOLDS:
        train, test = data.iloc[np.concatenate(folds[:fold - 1])], data.iloc[folds[fold - 1]]
        _assert_split(train, test)
        for variant in variants:
            p = score(variant, train, test)
            pred = pd.DataFrame({"repo": test.gh_project_name.to_numpy(), "build_id": test.tr_build_id.to_numpy(),
                                 "label": test.target_failed.to_numpy(), "probability": p,
                                 "new_breakage": test.prev_label.eq(0).to_numpy(), "fold": fold, "variant": variant,
                                 "label_policy": label_policy, "finish_buffer_hours": buffer})
            pred.to_parquet(PARTS / f"{label_policy}_{buffer}h_fold{fold}_{variant}.parquet", index=False)
            preds.append(pred)
            print(f"{label_policy} {buffer}h fold {fold} {variant} done", flush=True)
    return preds


def summarise(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, deltas = [], []
    for (policy, buffer, fold), unit in preds.groupby(["label_policy", "finish_buffer_hours", "fold"]):
        for subset in ("all", "new_breakage"):
            sub = unit if subset == "all" else unit[unit.new_breakage]
            per = {v: per_repository_metrics(g).set_index("repo") for v, g in sub.groupby("variant")}
            rng = np.random.default_rng(SEED + fold)
            for variant, m in per.items():
                g = sub[sub.variant.eq(variant)]
                row = {"label_policy": policy, "finish_buffer_hours": buffer, "fold": fold, "subset": subset,
                       "variant": variant, "runs": len(g), "failures": int(g.label.sum()),
                       "repositories_auc": int(m.within_repo_auc.notna().sum()),
                       "pooled_auc": roc_auc_score(g.label, g.probability) if g.label.nunique() == 2 else np.nan}
                for metric in METRICS:
                    v = m[metric].to_numpy(float)
                    row[metric] = float(np.nanmean(v))
                    row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_delta(v, rng)
                rows.append(row)
            rng = np.random.default_rng(SEED + 100 + fold)
            for variant, m in per.items():
                if variant == "C_rf":
                    continue
                repos = per["C_rf"].index.intersection(m.index)
                row = {"label_policy": policy, "finish_buffer_hours": buffer, "fold": fold, "subset": subset,
                       "comparison": f"{variant}_minus_C_rf", "repositories": len(repos)}
                for metric in METRICS:
                    diff = (m.loc[repos, metric] - per["C_rf"].loc[repos, metric]).to_numpy(float)
                    row[f"delta_{metric}"] = float(np.nanmean(diff))
                    row[f"delta_{metric}_ci_low"], row[f"delta_{metric}_ci_high"] = bootstrap_delta(diff, rng)
                deltas.append(row)
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def verdict(deltas: pd.DataFrame) -> dict:
    """The protocol's pre-registered decision rule for the primary result."""
    p = deltas[deltas.label_policy.eq("primary") & deltas.finish_buffer_hours.eq(0) & deltas.subset.eq("all")
               & deltas.comparison.eq("E_lgbmblend_minus_C_rf")]
    mean = float(p.delta_within_repo_auc.mean())
    excluding_zero = int((p.delta_within_repo_auc_ci_low > 0).sum())
    label = "transfers" if mean > 0 and excluding_zero >= 4 else "inconclusive" if mean > 0 else "does not transfer"
    return {"mean_delta_within_repo_auc": mean, "folds_with_ci_above_zero": excluding_zero, "folds": len(p), "verdict": label}


def main() -> None:
    PARTS.mkdir(exist_ok=True)
    manifest = {"protocol": PROTOCOL.name, "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                "settings": {f"{k[0]}_{k[1]}h": list(v) for k, v in SETTINGS.items()}, "data": {}}
    preds = []
    for policy in ("primary", "valid_only"):
        runs, counts = load_runs(policy)
        manifest["data"][policy] = counts
        for (p, buffer), variants in SETTINGS.items():
            if p == policy:
                preds += run_setting(policy, buffer, variants, runs)
    results, deltas = summarise(pd.concat(preds, ignore_index=True))
    results.to_csv(RESULTS, index=False); deltas.to_csv(DELTAS, index=False)
    manifest["primary_result"] = verdict(deltas)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(results.groupby(["label_policy", "finish_buffer_hours", "subset", "variant"])[["within_repo_auc", "pooled_auc"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
