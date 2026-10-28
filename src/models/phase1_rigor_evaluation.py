"""Versioned, build-level offline evaluation for the Phase-1 rigor study.

This module is deliberately isolated from the application and existing model
artifacts.  It writes only ``reports/phase1_v2_*`` outputs.  All choices below
are fixed in code before scoring: build-level units, chronological boundaries,
feature tiers, seed, calibration selection, and repository-cluster bootstrap.
"""
from __future__ import annotations

import argparse
import heapq
import json
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler

from src.inference.calibration import ProbabilityCalibrator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TRAVIS = ROOT / "data" / "processed" / "clean.parquet"
GHA_RAW = ROOT / "data" / "raw" / "final_research_dataset_MASTER.csv"
SEED = 20261013
BOOTSTRAPS = 1_000
ALERT_RATES = (0.10, 0.15)

# Tier A is the predeclared strict headline.  Tier B includes only fields whose
# point-in-time semantics are documented by TravisTorrent.  repo_age and
# repo_num_commits are intentionally absent: they are not described by the
# official field list for this release and require provenance before inclusion.
TIER_A_NUM = [
    "git_diff_src_churn", "git_diff_test_churn", "gh_diff_files_added",
    "gh_diff_files_deleted", "gh_diff_files_modified", "gh_diff_tests_added",
    "gh_diff_tests_deleted", "gh_diff_src_files", "gh_diff_doc_files",
    "gh_diff_other_files", "hour", "day_of_week", "month", "is_weekend",
    "is_night", "gh_is_pr", "gh_by_core_team_member", "is_master",
]
TIER_A_CAT = ["gh_lang"]
TIER_B_EXTRA = [
    "gh_team_size", "gh_num_commits_on_files_touched", "gh_sloc",
    "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
    "git_num_all_built_commits",
]
# Tier C is retained exactly for comparability.  Cplus is a pre-specified
# repository-history expansion; none of its values can use the focal build.
HISTORY = ["hist_prev_failed", "hist_repo_fail_rate_5", "hist_repo_fail_rate_all"]
HISTORY_PLUS = [
    "hist_repo_fail_rate_10", "hist_repo_fail_rate_20", "hist_failure_streak",
    "hist_hours_since_last_build", "hist_hours_since_last_failure",
]
# Fixed exploratory Tier D challenge set.  Every feature here is either
# available at trigger time or reconstructed solely from earlier completed
# builds.  These variants are reported together, not selected one by one.
SAME_COMMIT_HISTORY = ["hist_same_commit_completed_count", "hist_same_commit_fail_rate"]
# Availability audit found gh_num_commits_in_push and gh_pushed_at entirely
# missing in this export, so Tier D does not include all-missing columns.
CONTEXT = ["gh_description_complexity", "ctx_pr_age_hours"]
# git_diff_test_churn is identically zero in this export, so no within-repo
# normalization exists for it. Exclude it rather than let sklearn silently
# drop an all-missing derived column.
NORM_SOURCE = ["git_diff_src_churn", "gh_diff_files_added", "gh_diff_files_deleted", "gh_diff_files_modified", "gh_diff_tests_added", "gh_diff_tests_deleted", "gh_diff_src_files", "gh_diff_doc_files", "gh_diff_other_files"]
NORM = [f"norm_within_repo_{feature}" for feature in NORM_SOURCE]
# Tier E (frozen after fold-6-only development; folds 7-10 are confirmation).
# A "stream" is a PR (repo, PR number) or, for push builds, a (repo, branch).
# Outcomes enter only once the build derived-finished strictly before the
# current trigger; in-flight fields use only the existence of earlier starts.
STREAM = ["hist_stream_prev_failed", "hist_stream_fail_rate_5", "hist_stream_fail_rate_all", "hist_stream_completed",
          "hist_stream_streak", "hist_prevbuild_failed", "hist_repo_ewma"]
INFLIGHT = ["ctx_prevbuild_pending", "ctx_stream_pending", "ctx_repo_pending", "ctx_repo_started_1h",
            "ctx_hours_since_stream_start"]
# Build-matrix size is fixed by the config when Travis creates the build.
JOBS = ["ctx_job_count", "ctx_job_count_delta"]
EWMA_ALPHA = 0.2
RANK_MAX_GROUP = 5_000


def _rf() -> RandomForestClassifier:
    # Fixed, documented configuration; not chosen from these results.
    return RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1,
    )


def _travis_columns() -> list[str]:
    columns = list(dict.fromkeys([
        "tr_build_id", "tr_job_id", "tr_status", "tr_duration", "gh_build_started_at",
        "gh_project_name", "git_trigger_commit", "git_branch", *TIER_A_NUM, *TIER_A_CAT,
        *TIER_B_EXTRA, *CONTEXT[:1], "gh_pr_created_at", "git_prev_commit_resolution_status",
        "gh_pull_req_num", "tr_prev_build",
    ]))
    return [column for column in columns if column != "is_master"]


def load_travis_builds(finish_buffer_hours: int = 0) -> tuple[pd.DataFrame, dict]:
    """Collapse jobs, asserting every retained pre-build feature is constant."""
    jobs = pd.read_parquet(TRAVIS, columns=_travis_columns())
    jobs["gh_build_started_at"] = pd.to_datetime(jobs["gh_build_started_at"], utc=True, errors="raise")
    checked = [c for c in _travis_columns() if c not in {"tr_build_id", "tr_job_id", "tr_status", "tr_duration"}]
    nonconstant = []
    for col in checked:
        counts = jobs.groupby("tr_build_id", sort=False)[col].nunique(dropna=False)
        if (counts > 1).any():
            nonconstant.append(col)
    if nonconstant:
        raise AssertionError(f"Pre-build fields differ between jobs of a build: {nonconstant}")
    if finish_buffer_hours not in (0, 1, 6):
        raise ValueError("finish_buffer_hours must be one of 0, 1, 6")
    raw_duration = pd.to_numeric(jobs["tr_duration"], errors="coerce")
    valid_duration = raw_duration.where(raw_duration.gt(0))
    global_p95 = float(valid_duration.quantile(.95))
    repo_p95 = jobs.assign(_valid_duration=valid_duration).groupby("gh_project_name")._valid_duration.quantile(.95)
    fallback = jobs["gh_project_name"].map(repo_p95).fillna(global_p95)
    duration = raw_duration.where(raw_duration.gt(0), fallback)
    duration_quality = jobs.assign(
        duration_missing=raw_duration.isna(), duration_zero=raw_duration.eq(0),
        duration_negative=raw_duration.lt(0), duration_fallback=raw_duration.le(0) | raw_duration.isna(),
    ).groupby("gh_project_name", dropna=False)[["duration_missing", "duration_zero", "duration_negative", "duration_fallback"]].sum().reset_index()
    duration_quality["global_p95_seconds"] = global_p95
    duration_quality["repo_p95_seconds"] = duration_quality["gh_project_name"].map(repo_p95)
    REPORTS.mkdir(exist_ok=True)
    duration_quality.to_csv(REPORTS / "phase1_v2_duration_quality_by_repository.csv", index=False)
    jobs["_finished_at"] = jobs["gh_build_started_at"] + pd.to_timedelta(duration + 3600 * finish_buffer_hours, unit="s")
    agg = {c: "first" for c in checked if c != "gh_build_started_at"}
    agg.update({"gh_build_started_at": "min", "_finished_at": "max", "tr_status": lambda x: int(x.eq("failed").any()), "tr_job_id": "size"})
    builds = jobs.groupby("tr_build_id", sort=False, as_index=False).agg(agg).rename(columns={"tr_status": "target_failed", "tr_job_id": "job_count"})
    builds = builds.sort_values(["gh_build_started_at", "tr_build_id"], kind="mergesort").reset_index(drop=True)
    builds["is_master"] = builds["git_branch"].fillna("").astype(str).eq("master").astype("int8")
    for column in ("gh_pr_created_at",):
        builds[column]=pd.to_datetime(builds[column],utc=True,errors="coerce")
    for source, target in (("gh_pr_created_at","ctx_pr_age_hours"),):
        age=(builds["gh_build_started_at"]-builds[source]).dt.total_seconds()/3600
        builds[target]=age.where(age.ge(0))
        assert builds[target].dropna().ge(0).all(), f"negative pre-build age: {target}"
    totals = duration_quality[["duration_missing", "duration_zero", "duration_negative", "duration_fallback"]].sum().to_dict()
    return builds, {"job_rows": len(jobs), "builds": len(builds), "feature_constancy_assertion": "passed", "nonconstant_fields": nonconstant,
                    "duration_quality": {k:int(v) for k,v in totals.items()}, "global_duration_p95_seconds": global_p95,
                    "finish_buffer_hours": finish_buffer_hours}


def add_safe_history(builds: pd.DataFrame) -> pd.DataFrame:
    """Add history only after prior builds have *finished* before current start."""
    out = builds.copy()
    out[[*HISTORY, *HISTORY_PLUS, *SAME_COMMIT_HISTORY, *NORM]] = np.nan
    # No committer/account identity exists in clean.parquet; do not fabricate it.
    for _, idx in out.groupby("gh_project_name", sort=False).groups.items():
        idx = list(idx)
        # Equal finish times need an explicit, data-based order.  Never use
        # dataframe position: it can change after slicing/concatenation.
        pending: list[tuple[int, int, int, int, int]] = []
        completed_labels: list[int] = []
        failures = 0
        failure_streak = 0
        last_finished_ns: int | None = None
        last_failure_ns: int | None = None
        same_commit: dict[str, list[int]] = {}
        norm_count=np.zeros(len(NORM_SOURCE),dtype=int); norm_mean=np.zeros(len(NORM_SOURCE)); norm_m2=np.zeros(len(NORM_SOURCE))
        for row_index in idx:
            start = out.at[row_index, "gh_build_started_at"]
            start_ns = start.value
            while pending and pending[0][0] < start_ns:
                finished_ns, _start_ns, _build_id, label, prior_index = heapq.heappop(pending)
                completed_labels.append(label)
                failures += label
                failure_streak = failure_streak + 1 if label else 0
                last_finished_ns = finished_ns
                if label:
                    last_failure_ns = finished_ns
                sha=out.at[prior_index,"git_trigger_commit"]
                if pd.notna(sha):
                    key=str(sha); state=same_commit.setdefault(key,[0,0]); state[0]+=1; state[1]+=label
                for position, feature in enumerate(NORM_SOURCE):
                    value=pd.to_numeric(out.at[prior_index,feature],errors="coerce")
                    if pd.notna(value):
                        transformed=np.log1p(max(float(value),0.0)); norm_count[position]+=1
                        delta=transformed-norm_mean[position]; norm_mean[position]+=delta/norm_count[position]; norm_m2[position]+=delta*(transformed-norm_mean[position])
            if completed_labels:
                out.at[row_index, "hist_prev_failed"] = completed_labels[-1]
                out.at[row_index, "hist_repo_fail_rate_all"] = failures / len(completed_labels)
                for window in (5, 10, 20):
                    out.at[row_index, f"hist_repo_fail_rate_{window}"] = float(np.mean(completed_labels[-window:]))
                out.at[row_index, "hist_failure_streak"] = failure_streak
                out.at[row_index, "hist_hours_since_last_build"] = (start_ns - last_finished_ns) / 3.6e12
                if last_failure_ns is not None:
                    out.at[row_index, "hist_hours_since_last_failure"] = (start_ns - last_failure_ns) / 3.6e12
            current_sha=out.at[row_index,"git_trigger_commit"]
            if pd.notna(current_sha) and str(current_sha) in same_commit:
                count, failed=same_commit[str(current_sha)]
                out.at[row_index,"hist_same_commit_completed_count"]=count
                out.at[row_index,"hist_same_commit_fail_rate"]=failed/count
            for position, feature in enumerate(NORM_SOURCE):
                value=pd.to_numeric(out.at[row_index,feature],errors="coerce")
                if norm_count[position] >= 2 and pd.notna(value):
                    std=np.sqrt(norm_m2[position]/(norm_count[position]-1))
                    if std > 0:
                        out.at[row_index,NORM[position]]=(np.log1p(max(float(value),0.0))-norm_mean[position])/std
            heapq.heappush(pending, (out.at[row_index, "_finished_at"].value, start_ns, int(out.at[row_index, "tr_build_id"]), int(out.at[row_index, "target_failed"]), int(row_index)))
    return out


def _stream_keys(d: pd.DataFrame) -> list[str]:
    is_pr = d.gh_is_pr.fillna(False).astype(bool).to_numpy(); number = d.gh_pull_req_num.to_numpy()
    branch = d.git_branch.fillna("").astype(str).to_numpy()
    return [f"pr{number[i]}" if is_pr[i] and pd.notna(number[i]) else f"br{branch[i]}" for i in range(len(d))]


def add_stream_history(builds: pd.DataFrame) -> pd.DataFrame:
    """Tier E: stream-level history and in-flight context, finish-time safe."""
    out = builds.copy(); n = len(out)
    cols = {c: np.full(n, np.nan) for c in [*STREAM, *INFLIGHT, *JOBS]}
    start = out.gh_build_started_at.dt.as_unit("ns").astype("int64").to_numpy(); finish = out._finished_at.dt.as_unit("ns").astype("int64").to_numpy()
    build_id = out.tr_build_id.to_numpy(np.int64); label = out.target_failed.to_numpy(np.int64)
    prev = out.tr_prev_build.to_numpy(); jobs = out.job_count.to_numpy(float); keys = _stream_keys(out)
    hour_ns = 3_600_000_000_000
    for _, idx in out.groupby("gh_project_name", sort=False).indices.items():
        pending: list[tuple[int, int, int, int, int]] = []
        done: dict[int, int] = {}; started: set[int] = set()
        stream_labels: dict[str, list[int]] = {}; stream_streak: dict[str, int] = {}; stream_last_start: dict[str, tuple[int, float, int]] = {}
        recent: deque[int] = deque(); ewma: float | None = None
        for i in idx:
            t0 = start[i]
            while pending and pending[0][0] < t0:
                _finish, _start, prior_id, prior_label, j = heapq.heappop(pending)
                done[prior_id] = prior_label; k = keys[j]
                stream_labels.setdefault(k, []).append(prior_label)
                stream_streak[k] = stream_streak.get(k, 0) + 1 if prior_label else 0
                ewma = float(prior_label) if ewma is None else EWMA_ALPHA * prior_label + (1 - EWMA_ALPHA) * ewma
            k = keys[i]; history = stream_labels.get(k)
            cols["hist_stream_completed"][i] = len(history) if history else 0
            if history:
                cols["hist_stream_prev_failed"][i] = history[-1]
                cols["hist_stream_fail_rate_5"][i] = np.mean(history[-5:])
                cols["hist_stream_fail_rate_all"][i] = np.mean(history)
                cols["hist_stream_streak"][i] = stream_streak[k]
            if ewma is not None: cols["hist_repo_ewma"][i] = ewma
            if pd.notna(prev[i]):
                p = int(prev[i])
                if p in done: cols["hist_prevbuild_failed"][i] = done[p]; cols["ctx_prevbuild_pending"][i] = 0
                elif p in started: cols["ctx_prevbuild_pending"][i] = 1
            last = stream_last_start.get(k)
            if last is not None and last[0] < t0:
                cols["ctx_hours_since_stream_start"][i] = (t0 - last[0]) / 3.6e12
                cols["ctx_stream_pending"][i] = 0 if last[2] in done else 1
                cols["ctx_job_count_delta"][i] = jobs[i] - last[1]
            cols["ctx_repo_pending"][i] = len(pending)
            while recent and recent[0] < t0 - hour_ns: recent.popleft()
            cols["ctx_repo_started_1h"][i] = len(recent); cols["ctx_job_count"][i] = jobs[i]
            recent.append(t0); started.add(int(build_id[i])); stream_last_start[k] = (t0, jobs[i], int(build_id[i]))
            heapq.heappush(pending, (finish[i], t0, int(build_id[i]), int(label[i]), i))
    for column, values in cols.items(): out[column] = values
    return out


def _assert_stream_history(data: pd.DataFrame, repos: int = 40) -> None:
    """Brute-force recomputation of Tier E outcome fields on sampled repositories."""
    sizes = data.gh_project_name.value_counts(); pool = sizes[(sizes >= 50) & (sizes <= 800)].index
    for repo in np.random.default_rng(SEED).choice(pool, size=min(repos, len(pool)), replace=False):
        d = data.loc[data.gh_project_name.eq(repo)]; key = np.array(_stream_keys(d))
        finish = d._finished_at.dt.as_unit("ns").astype("int64").to_numpy(); start = d.gh_build_started_at.dt.as_unit("ns").astype("int64").to_numpy()
        ids = d.tr_build_id.to_numpy(np.int64); y = d.target_failed.to_numpy()
        for r in range(len(d)):
            row = d.iloc[r]; mask = (key == key[r]) & (finish < start[r])
            order = np.lexsort((ids[mask], start[mask], finish[mask])); prior = y[mask][order]
            if len(prior):
                assert row.hist_stream_prev_failed == prior[-1], f"unsafe stream previous outcome for {repo}"
                assert np.isclose(row.hist_stream_fail_rate_5, prior[-5:].mean()), f"unsafe stream 5-build rate for {repo}"
                assert np.isclose(row.hist_stream_fail_rate_all, prior.mean()), f"unsafe stream rate for {repo}"
            else:
                assert pd.isna(row.hist_stream_prev_failed), f"stream history before completion for {repo}"
            if pd.notna(row.tr_prev_build):
                hit = np.flatnonzero(ids == int(row.tr_prev_build))
                if len(hit) and finish[hit[0]] < start[r]: assert row.hist_prevbuild_failed == y[hit[0]], f"unsafe previous-build outcome for {repo}"
                else: assert pd.isna(row.hist_prevbuild_failed), f"previous-build outcome before completion for {repo}"


def tier_features(tier: str) -> tuple[list[str], list[str]]:
    if tier == "A": return TIER_A_NUM, TIER_A_CAT
    if tier == "B": return [*TIER_A_NUM, *TIER_B_EXTRA], TIER_A_CAT
    if tier == "C": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY], TIER_A_CAT
    if tier == "Cplus": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *HISTORY_PLUS], TIER_A_CAT
    if tier == "Dcommit": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *SAME_COMMIT_HISTORY], TIER_A_CAT
    if tier == "Dcontext": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *CONTEXT], TIER_A_CAT
    if tier == "Dnorm": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *NORM], TIER_A_CAT
    if tier == "Dall": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *SAME_COMMIT_HISTORY, *CONTEXT, *NORM], TIER_A_CAT
    if tier == "DnormV2": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *NORM], TIER_A_CAT
    if tier == "DallV2": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *SAME_COMMIT_HISTORY, *CONTEXT, *NORM], TIER_A_CAT
    if tier == "E": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *STREAM, *INFLIGHT, *JOBS], TIER_A_CAT
    if tier == "Enojobs": return [*TIER_A_NUM, *TIER_B_EXTRA, *HISTORY, *STREAM, *INFLIGHT], TIER_A_CAT
    raise ValueError(tier)


def _lgbm_params() -> dict:
    # Fixed before confirmation folds were scored; flat across fold-6 variants.
    return dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=SEED, verbose=-1)


def _lgbm_frame(d: pd.DataFrame, numeric: list[str], categorical: list[str], categories: dict | None = None) -> pd.DataFrame:
    x = d[numeric + categorical].copy()
    for c in categorical:
        x[c] = pd.Categorical(x[c], categories=categories[c]) if categories else x[c].astype("category")
    return x


def _rank_groups(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row order and group sizes for repo x calendar-month ranking queries.

    Queries without both outcomes carry no ranking signal and are dropped;
    queries above LightGBM's row limit are cut into consecutive chunks.
    """
    month = train.gh_build_started_at.dt.tz_localize(None).dt.to_period("M").astype(str).to_numpy()
    base = train.gh_project_name.astype(str).to_numpy() + "|" + month
    order = np.argsort(base, kind="stable")
    _, query = np.unique(base[order], return_inverse=True)
    position = pd.Series(np.arange(len(order))).groupby(query).cumcount().to_numpy()
    query = query * 1_000 + position // RANK_MAX_GROUP
    y = train.target_failed.to_numpy()[order]
    mixed = pd.Series(y).groupby(query).transform("nunique").to_numpy() == 2
    return order[mixed], np.unique(query[mixed], return_counts=True)[1]


def fit_predict_lgbm(kind: str, train: pd.DataFrame, test: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    """lgbm: binary; lgbmrank: rank_xendcg by repo-month; lgbmblend: equal z-blend of both.

    Blend standardisation uses training-score moments only, never test scores.
    """
    import lightgbm as lgb
    x = _lgbm_frame(train, numeric, categorical); categories = {c: x[c].cat.categories for c in categorical}
    xt = _lgbm_frame(test, numeric, categorical, categories); scores = {}
    if kind in {"lgbm", "lgbmblend"}:
        clf = lgb.LGBMClassifier(**_lgbm_params(), objective="binary").fit(x, train.target_failed)
        scores["binary"] = (_logit(clf.predict_proba(x)[:, 1]), _logit(clf.predict_proba(xt)[:, 1]))
        if kind == "lgbm": return clf.predict_proba(xt)[:, 1]
    if kind in {"lgbmrank", "lgbmblend"}:
        order, sizes = _rank_groups(train)
        ranker = lgb.LGBMRanker(**_lgbm_params(), objective="rank_xendcg").fit(x.iloc[order], train.target_failed.to_numpy()[order], group=sizes)
        scores["rank"] = (ranker.predict(x), ranker.predict(xt))
    if kind == "lgbmrank": train_score, test_score = scores["rank"]
    elif kind == "lgbmblend":
        z = lambda pair, ref: (pair - ref.mean()) / ref.std()
        train_score = 0.5 * z(scores["rank"][0], scores["rank"][0]) + 0.5 * z(scores["binary"][0], scores["binary"][0])
        test_score = 0.5 * z(scores["rank"][1], scores["rank"][0]) + 0.5 * z(scores["binary"][1], scores["binary"][0])
    else: raise ValueError(kind)
    # One global increasing map to probabilities; AUCs are unchanged by it.
    platt = LogisticRegression(max_iter=500).fit(train_score.reshape(-1, 1), train.target_failed)
    assert platt.coef_[0, 0] > 0, "score-to-probability map must be increasing"
    return platt.predict_proba(test_score.reshape(-1, 1))[:, 1]


def pipeline(kind: str, numeric: list[str], categorical: list[str]):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    if kind == "majority": model = DummyClassifier(strategy="most_frequent")
    elif kind == "logistic": model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    elif kind == "rf": model = _rf()
    else: raise ValueError(kind)
    return Pipeline([("preprocess", prep), ("model", model)])


def fit_model(kind: str, train: pd.DataFrame, numeric: list[str], categorical: list[str]):
    """Fit a fixed model variant; balanced uses inverse training repo size."""
    model=pipeline("rf" if kind == "balanced_rf" else kind,numeric,categorical)
    if kind == "balanced_rf":
        counts=train.groupby("gh_project_name").gh_project_name.transform("size")
        weights=(1/counts).to_numpy(float)
        weights=weights/weights.mean()
        return model.fit(train[numeric+categorical],train.target_failed,model__sample_weight=weights)
    return model.fit(train[numeric+categorical],train.target_failed)


def fit_project_resampled_rf(train: pd.DataFrame, numeric: list[str], categorical: list[str]):
    """Fit the online per-project RF with training-only random oversampling.

    RandomOverSampler is used instead of synthetic interpolation because this
    mix of one-hot categories and sparse per-project folds makes SMOTE's
    neighbourhood assumptions fragile.  It never sees a test-fold row.
    """
    base=pipeline("rf",numeric,categorical)
    preprocess=base.named_steps["preprocess"]
    x_train=preprocess.fit_transform(train[numeric+categorical])
    before=train.target_failed.value_counts().to_dict()
    sampler=RandomOverSampler(random_state=SEED)
    x_resampled,y_resampled=sampler.fit_resample(x_train,train.target_failed)
    # Oversampling already balances labels; retaining class_weight='balanced'
    # would double-correct the imbalance.
    estimator=_rf().set_params(class_weight=None).fit(x_resampled,y_resampled)
    return preprocess,estimator,{"sampler":"RandomOverSampler","sampling_scope":"training_fold_only","class_counts_before":before,"class_counts_after":pd.Series(y_resampled).value_counts().to_dict()}


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((bucket == b).mean() * abs(y[bucket == b].mean() - p[bucket == b].mean()) for b in range(bins) if (bucket == b).any()))


def _top_metrics(y: np.ndarray, p: np.ndarray, rate: float) -> dict:
    n = max(1, int(np.ceil(len(y) * rate)))
    chosen = np.argsort(-p, kind="mergesort")[:n]
    precision = float(y[chosen].mean())
    recall = float(y[chosen].sum() / y.sum()) if y.sum() else np.nan
    return {f"precision_at_{int(rate*100)}": precision, f"recall_at_{int(rate*100)}": recall, f"lift_at_{int(rate*100)}": precision / y.mean() if y.mean() else np.nan}


def metric_row(y, p, threshold: float, model: str, tier: str, split: str) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    row = {"dataset": "travis", "split": split, "tier": tier, "model": model, "n": len(y), "failures": int(y.sum()), "prevalence": float(y.mean()),
           "roc_auc": float(roc_auc_score(y,p)) if np.unique(y).size == 2 else np.nan,
           "pr_auc": float(average_precision_score(y,p)) if y.sum() else np.nan,
           "precision": float(precision), "recall": float(recall), "f1": float(f1),
           "brier": float(brier_score_loss(y,p)), "ece": _ece(y,p), "threshold": float(threshold)}
    for r in ALERT_RATES: row.update(_top_metrics(y,p,r))
    return row


def weekly_block_ci(pred: pd.DataFrame, threshold: float, seed: int = SEED, repetitions: int = BOOTSTRAPS) -> dict:
    """Percentile CI resampling whole ISO weeks within one repository."""
    # Weeks are defined after conversion to UTC; make the timezone removal
    # explicit so pandas does not emit a misleading conversion warning.
    d=pred.copy(); d["week"]=pd.to_datetime(d.trigger_time,utc=True).dt.tz_localize(None).dt.to_period("W").astype(str)
    weeks=d.week.unique(); rng=np.random.default_rng(seed); values={"roc_auc":[],"recall_at_15":[],"lift_at_15":[]}
    if len(weeks)<2: return {f"weekly_block_{m}_{q}":np.nan for m in values for q in ("low","high")}
    for _ in range(repetitions):
        sample=pd.concat([d.loc[d.week.eq(w)] for w in rng.choice(weeks,size=len(weeks),replace=True)],ignore_index=True)
        if sample.label.nunique()!=2: continue
        row=metric_row(sample.label,sample.probability,threshold,"x","x","x")
        for name in values: values[name].append(row[name])
    return {f"weekly_block_{name}_{q}":float(np.nanpercentile(value, 2.5 if q=="low" else 97.5)) if value else np.nan for name,value in values.items() for q in ("low","high")}


def paired_weekly_delta_ci(reference: pd.DataFrame, candidate: pd.DataFrame, threshold: float, seed: int = SEED, repetitions: int = BOOTSTRAPS) -> dict:
    """Paired weekly-block CI for candidate minus cold-start ranking metrics."""
    left=reference.copy(); right=candidate.copy()
    left["week"]=pd.to_datetime(left.trigger_time,utc=True).dt.tz_localize(None).dt.to_period("W").astype(str)
    right["week"]=pd.to_datetime(right.trigger_time,utc=True).dt.tz_localize(None).dt.to_period("W").astype(str)
    weeks=left.week.unique(); rng=np.random.default_rng(seed); values={"roc_auc":[],"recall_at_15":[],"lift_at_15":[]}
    for _ in range(repetitions):
        sample_weeks=rng.choice(weeks,size=len(weeks),replace=True)
        a=pd.concat([left.loc[left.week.eq(w)] for w in sample_weeks],ignore_index=True)
        b=pd.concat([right.loc[right.week.eq(w)] for w in sample_weeks],ignore_index=True)
        if a.label.nunique()!=2: continue
        ma=metric_row(a.label,a.probability,threshold,"x","x","x"); mb=metric_row(b.label,b.probability,threshold,"x","x","x")
        for name in values: values[name].append(mb[name]-ma[name])
    return {f"paired_weekly_delta_{name}_{q}":float(np.nanpercentile(value,2.5 if q=="low" else 97.5)) if value else np.nan for name,value in values.items() for q in ("low","high")}


def cluster_ci(pred: pd.DataFrame, point: dict) -> dict:
    """Repository-cluster percentile bootstrap; rows remain intact within clusters."""
    rng = np.random.default_rng(SEED)
    groups = list(pred["repo"].dropna().unique())
    metrics = ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier", "ece", "precision_at_10", "recall_at_10", "lift_at_10", "precision_at_15", "recall_at_15", "lift_at_15"]
    values = {m: [] for m in metrics}
    if len(groups) < 2: return {f"{m}_ci_low": np.nan for m in metrics} | {f"{m}_ci_high": np.nan for m in metrics}
    for _ in range(BOOTSTRAPS):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([pred.loc[pred.repo.eq(g)] for g in chosen], ignore_index=True)
        r = metric_row(sample.y, sample.p, point["threshold"], point["model"], point["tier"], point["split"])
        for m in metrics: values[m].append(r[m])
    return {f"{m}_ci_low": float(np.nanpercentile(v, 2.5)) for m,v in values.items()} | {f"{m}_ci_high": float(np.nanpercentile(v,97.5)) for m,v in values.items()}


def calibrate(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, numeric, categorical):
    """Predeclared rule: choose minimum validation Brier; ties raw > sigmoid > isotonic."""
    base = pipeline("rf", numeric, categorical).fit(train[numeric+categorical], train.target_failed)
    pv = base.predict_proba(val[numeric+categorical])[:,1]
    candidates = {m: ProbabilityCalibrator.fit(m, pv, val.target_failed) for m in ("raw", "sigmoid", "isotonic")}
    order = {"raw":0,"sigmoid":1,"isotonic":2}
    scored = sorted(((brier_score_loss(val.target_failed, c.transform(pv)), order[m], m, c) for m,c in candidates.items()))
    _,_,chosen,calibrator = scored[0]
    pv2, pt = calibrator.transform(pv), calibrator.transform(base.predict_proba(test[numeric+categorical])[:,1])
    threshold = float(np.sort(pv2)[max(0, int(np.ceil(.85*len(pv2)))-1)])
    return base, calibrator, chosen, threshold, pt, float(brier_score_loss(val.target_failed,pv2)), float(val.target_failed.mean())


def evaluate_travis() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data, unit = load_travis_builds()
    data = add_safe_history(data)
    n = len(data); train_end=int(n*.72); val_end=int(n*.80)
    train, val, test = data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]
    rows=[]; prediction_sets=[]; calibration_info={}
    # Headline tiers, no test-driven selection.  Majority/logistic/RF all report
    # at their predeclared 0.5 decision threshold; RF additionally is calibrated.
    for tier in ("A","B","C"):
        numeric,categorical=tier_features(tier)
        for kind in ("majority","logistic","rf"):
            m=pipeline(kind,numeric,categorical).fit(train[numeric+categorical],train.target_failed)
            p=m.predict_proba(test[numeric+categorical])[:,1]
            r=metric_row(test.target_failed,p,.5,kind,tier,"final_20pct_development_test")
            pred=pd.DataFrame({"repo":test.gh_project_name,"y":test.target_failed,"p":p})
            r.update(cluster_ci(pred,r)); rows.append(r); prediction_sets.append((tier,kind,pred,r))
        # calibration is reported only for the RF tier, selected solely by validation Brier.
        _m,_c,chosen,threshold,p,brier_val,prev_val=calibrate(train,val,test,numeric,categorical)
        r=metric_row(test.target_failed,p,threshold,"rf_calibrated",tier,"final_20pct_development_test")
        r.update({"calibrator":chosen,"validation_brier":brier_val,"validation_constant_brier":prev_val*(1-prev_val),"validation_brier_skill":1-brier_val/(prev_val*(1-prev_val))})
        pred=pd.DataFrame({"repo":test.gh_project_name,"y":test.target_failed,"p":p}); r.update(cluster_ci(pred,r)); rows.append(r); prediction_sets.append((tier,"rf_calibrated",pred,r))
        calibration_info[tier]={"chosen_by_validation_brier":chosen,"threshold":threshold,"validation_brier":brier_val,"validation_prevalence":prev_val}
    # History-only baselines are reconstructed, not raw linkage encodings.
    for feature, name in [("hist_prev_failed","previous_status_reconstructed"),("hist_repo_fail_rate_5","last_5_failure_rate")]:
        p=test[feature].fillna(train.target_failed.mean()).to_numpy(float)
        r=metric_row(test.target_failed,p,.5,name,"C","final_20pct_development_test")
        pred=pd.DataFrame({"repo":test.gh_project_name,"y":test.target_failed,"p":p}); r.update(cluster_ci(pred,r)); rows.append(r)
    # Five expanding windows: folds 6..10, train all earlier folds.  Tier A RF.
    folds=np.array_split(np.arange(n),10)
    numeric,categorical=tier_features("A")
    for j in range(5,10):
        tr=data.iloc[np.concatenate(folds[:j])]; te=data.iloc[folds[j]]
        m=pipeline("rf",numeric,categorical).fit(tr[numeric+categorical],tr.target_failed)
        p=m.predict_proba(te[numeric+categorical])[:,1]
        r=metric_row(te.target_failed,p,.5,"rf","A",f"walk_forward_fold_{j+1}")
        r.update(cluster_ci(pd.DataFrame({"repo":te.gh_project_name,"y":te.target_failed,"p":p}),r)); rows.append(r)
    # Per-project online evaluation: >=20 total failures; report all valid folds.
    proj=[]
    numeric,categorical=tier_features("A")
    eligible=data.groupby("gh_project_name").target_failed.agg(["size","sum"]).query("sum >= 20").index
    for repo in eligible:
        d=data.loc[data.gh_project_name.eq(repo)].sort_values("gh_build_started_at",kind="mergesort").reset_index(drop=True)
        folds=np.array_split(np.arange(len(d)),10); aucs=[]; f1s=[]; tested=0
        for j in range(5,10):
            tr,te=d.iloc[np.concatenate(folds[:j])],d.iloc[folds[j]]
            if tr.target_failed.sum()==0 or te.target_failed.sum()==0: continue
            m=pipeline("rf",numeric,categorical).fit(tr[numeric+categorical],tr.target_failed); p=m.predict_proba(te[numeric+categorical])[:,1]
            aucs.append(roc_auc_score(te.target_failed,p)); f1s.append(metric_row(te.target_failed,p,.5,"rf","A","x")["f1"]); tested+=1
        proj.append({"repository":repo,"builds":len(d),"failures":int(d.target_failed.sum()),"prevalence":float(d.target_failed.mean()),"eligible":True,"valid_test_folds":tested,"mean_auc":np.mean(aucs) if aucs else np.nan,"mean_f1":np.mean(f1s) if f1s else np.nan})
    # Within-repository AUC on final test, based on headline A calibrated output.
    headline=next(p for t,k,p,r in prediction_sets if t=="A" and k=="rf_calibrated")
    within=[]
    for repo,d in headline.groupby("repo"):
        if d.y.nunique()==2: within.append(roc_auc_score(d.y,d.p))
    calibration_info["within_repo_auc_macro_final_test"] = float(np.mean(within))
    return pd.DataFrame(rows), pd.DataFrame(proj), {**unit, "calibration":calibration_info, "history_note":"committer fail history unavailable: no committer identifier in processed Travis data", "prev_status_raw_categories": sorted(data.git_prev_commit_resolution_status.dropna().astype(str).unique().tolist())}


GHA_NUM = ["run_number", "msg_len", "is_merge", "num_parents", "additions", "deletions", "total_churn", "files_modified", "time_since_last_commit", "hour", "day_of_week", "month", "is_weekend", "is_night"]
GHA_CAT = ["event", "head_branch"]


def load_gha(first_attempt_only: bool) -> pd.DataFrame:
    raw_numeric=[column for column in GHA_NUM if column not in {"hour","day_of_week","month","is_weekend","is_night"}]
    columns = ["run_id", "repo", "workflow_id", "created_at", "conclusion", "run_attempt", *raw_numeric, *GHA_CAT]
    raw = pd.read_csv(GHA_RAW, usecols=columns, low_memory=False)
    raw["created_at"] = pd.to_datetime(raw.created_at, utc=True, errors="coerce")
    raw["conclusion"] = raw.conclusion.astype("string").str.lower().str.strip()
    d = raw.loc[raw.workflow_id.notna() & raw.created_at.notna() & raw.conclusion.isin(["success","failure","startup_failure"])].copy()
    if first_attempt_only: d=d.loc[d.run_attempt.eq(1)]
    d["target_failed"] = d.conclusion.isin(["failure","startup_failure"]).astype(int)
    d["is_merge"] = d.is_merge.astype(float)
    d["hour"] = d.created_at.dt.hour; d["day_of_week"] = d.created_at.dt.dayofweek; d["month"] = d.created_at.dt.month
    d["is_weekend"] = d.day_of_week.isin([5,6]).astype(int); d["is_night"] = d.hour.between(0,5).astype(int)
    return d.sort_values(["created_at","run_id"],kind="mergesort").reset_index(drop=True)


def evaluate_gha() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]; lroo=[]
    for first in (True,False):
        d=load_gha(first); cut=int(len(d)*.8); train,test=d.iloc[:cut],d.iloc[cut:]
        tag="first_attempt_only" if first else "all_attempts_sensitivity"
        for kind in ("majority","logistic","rf"):
            m=pipeline(kind,GHA_NUM,GHA_CAT).fit(train[GHA_NUM+GHA_CAT],train.target_failed); p=m.predict_proba(test[GHA_NUM+GHA_CAT])[:,1]
            r=metric_row(test.target_failed,p,.5,kind,"GHA",f"gha_{tag}_temporal_80_20"); r["dataset"]="gha"; r.update(cluster_ci(pd.DataFrame({"repo":test.repo,"y":test.target_failed,"p":p}),r)); rows.append(r)
        # All eligible GHA repositories: >=200 later rows and >=15 later failures, fixed pre-fit rule.
        eligible=test.groupby("repo").target_failed.agg(builds="size",failures="sum").query("builds >= 200 and failures >= 15")
        excluded=test.groupby("repo").target_failed.agg(builds="size",failures="sum").reset_index().assign(eligible=lambda x:(x.builds>=200)&(x.failures>=15))
        excluded.to_csv(REPORTS/f"phase1_v2_gha_{tag}_holdout_eligibility.csv",index=False)
        for repo in eligible.index:
            tr=train.loc[train.repo.ne(repo)]; te=test.loc[test.repo.eq(repo)]
            m=pipeline("rf",GHA_NUM,GHA_CAT).fit(tr[GHA_NUM+GHA_CAT],tr.target_failed); p=m.predict_proba(te[GHA_NUM+GHA_CAT])[:,1]
            r=metric_row(te.target_failed,p,.5,"rf","GHA",f"gha_loro_{tag}"); r["dataset"]="gha"; r["repository"]=repo; r["eligibility_rule"]="later test builds >=200 and failures >=15"; lroo.append(r)
    return pd.DataFrame(rows),pd.DataFrame(lroo)


PARTS = REPORTS / "phase1_v2_parts"


def _record_skip(unit: str, reason: str, finish_buffer_hours: int) -> None:
    path=REPORTS / "phase1_v2_skipped_units.csv"
    row=pd.DataFrame([{"unit":unit,"finish_buffer_hours":finish_buffer_hours,"reason":reason,"timestamp_utc":pd.Timestamp.now(tz="UTC")}])
    row.to_csv(path,index=False,mode="a",header=not path.exists())


def _manifest() -> dict:
    """Fixed run configuration, written before any unit fits."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    import sklearn
    return {"seed": SEED, "bootstrap_repetitions": BOOTSTRAPS,
            "rf": {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5,
                   "min_samples_leaf": 2, "class_weight": "balanced", "n_jobs": -1},
            "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "platform": platform.platform(),
            "git_commit": commit,
            "build_finish_time": "max over jobs of (gh_build_started_at + valid positive duration or repo p95/global p95 fallback + finish buffer)",
            "finish_buffer_hours": {"headline_tier_c": 0, "sensitivity": [1, 6], "allowed": [0, 1, 6]},
            "flagged_repo_identity_features_for_ablation": ["gh_lang", "gh_team_size", "gh_sloc", "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc", "gh_num_commits_on_files_touched", "git_num_all_built_commits", "gh_repo_age", "gh_repo_num_commits", "tr_build_number", "hist_prev_failed", "hist_repo_fail_rate_5", "hist_repo_fail_rate_all"],
            "history_rule": "only same-repository builds whose derived finish time is strictly earlier than current trigger/start time; ties among completed builds are ordered by finish time, trigger time, then build id",
            "tier_cplus_history_features": HISTORY_PLUS,
            "tier_d_exploratory_challenge": {
                "predeclared_variants": ["Dcommit", "Dcontext", "DnormV2", "DallV2", "C_with_repository_balanced_rf"],
                "evaluation": "same five expanding walk-forward folds; macro within-repository metrics primary; report every variant",
                "same_commit_rule": "prior build of same trigger SHA must have derived finish strictly before current trigger",
                "normalization_rule": "log1p feature z-score against same-repository builds that derived-finished strictly before current trigger",
                "balanced_rf_rule": "inverse training-repository build count, normalized to mean weight one",
                "availability_exclusions": ["committer identity", "file paths", "gh_num_commits_in_push (all missing)", "gh_pushed_at (all missing)"],
                "normalization_correction": "Dnorm preliminary parts retained but superseded by DnormV2 because git_diff_test_churn is identically zero and its derived z-score is undefined",
            },
            "tier_e_protocol": {
                "development_fold": 6, "confirmation_folds": [7, 8, 9, 10],
                "frozen_before_confirmation": ["STREAM/INFLIGHT/JOBS feature lists", "LightGBM params", "rank_xendcg objective with repo x calendar-month queries", "equal z-blend of ranker and binary LightGBM using training-score moments", "global Platt map fit on training scores"],
                "stream_definition": "PR builds: (repo, gh_pull_req_num); push builds: (repo, git_branch)",
                "outcome_rule": "stream and tr_prev_build outcomes used only if that build derived-finished strictly before current trigger",
                "inflight_rule": "counts and flags use only the existence/start time of earlier-started builds, never their outcome",
                "job_count_rule": "build-matrix size fixed at build creation; reported with and without (Enojobs) for provenance",
                "verification": "_assert_stream_history brute-force recomputation on 40 sampled repositories every walk run",
                "fold6_development_variants_tried": ["RF/LGBM binary on C, C+S, C+S+I, C+S+I+J", "repo-size weights", "identity-feature removal", "lambdarank vs rank_xendcg", "week/month/quarter queries", "4 LightGBM hyperparameter settings", "5-seed bagging", "ranker/binary blend weights 0.25 and 0.5", "global label-shuffle null (3 seeds, 0.48-0.50)"],
            },
            "walk_tier_c_shuffle_control": {"fold": 8, "seeds": 10, "label_permutation": "global training labels only", "purpose": "additional null distribution; no model selection"},
            "random_score_sanity_control": {"fold": 8, "draws": 1000, "scores": "independent uniform random scores over saved fixed test predictions", "expected": "macro within-repository AUC centered near 0.5"},
            "per_project_online_protocol": {"folds": 10, "test_folds": [6,7,8,9,10], "training_folds_for_j":"1..j-1", "model":"200-tree RF", "resampling":"RandomOverSampler on transformed training fold only; no test resampling", "skip_rule":"skip a repository-fold with no failed build in train or test"},
            "unavailable_requested_features": {
                "committer_history": "clean.parquet contains no stable committer/account identifier",
                "file_level_failure_history": "clean.parquet contains aggregate diff counts but no changed-file paths",
                "change_type_indicators": "clean.parquet contains no file paths or file-type flags from which CI/config/dependency changes can be derived",
            }}


def _write_manifest_with_duration_quality() -> None:
    manifest=_manifest(); quality=REPORTS / "phase1_v2_duration_quality_by_repository.csv"
    if quality.exists():
        d=pd.read_csv(quality)
        manifest["duration_quality"]={k:int(v) for k,v in d[["duration_missing","duration_zero","duration_negative","duration_fallback"]].sum().items()}
        manifest["duration_fallback"]={"invalid_definition":"missing, zero, or negative tr_duration", "repository_fallback":"positive-duration p95 within repository", "global_fallback":"positive-duration p95 if repository p95 unavailable", "global_p95_seconds":float(d.global_p95_seconds.iloc[0])}
    (REPORTS / "phase1_v2_run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def _part_key(unit: str, finish_buffer_hours: int) -> str:
    return f"{unit}__finish_buffer_hours_{finish_buffer_hours}"


def _save_part(unit: str, row: dict, predictions: pd.DataFrame, elapsed: float, finish_buffer_hours: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    key=_part_key(unit, finish_buffer_hours)
    row = {**row, "unit": unit, "finish_buffer_hours": finish_buffer_hours, "elapsed_seconds": elapsed}
    pd.DataFrame([row]).to_json(PARTS / f"{key}.json", orient="records", indent=2)
    predictions.to_parquet(PARTS / f"{key}.parquet", index=False)


def _load_part(unit: str, finish_buffer_hours: int) -> bool:
    key=_part_key(unit, finish_buffer_hours)
    return (PARTS / f"{key}.json").exists() and (PARTS / f"{key}.parquet").exists()


def _add_loro_adapter_comparisons(checkpoint_unit: str, finish_buffer_hours: int) -> None:
    """Persist paired adapter-minus-baseline effects from saved predictions.

    This intentionally never refits a model or resamples individual rows: the
    identical weekly blocks are resampled for both saved score vectors.
    """
    adapter_key=_part_key(f"{checkpoint_unit}_adapter", finish_buffer_hours)
    adapter_json=PARTS / f"{adapter_key}.json"
    adapter_pred=pd.read_parquet(PARTS / f"{adapter_key}.parquet")
    row=json.loads(adapter_json.read_text(encoding="utf-8"))[0]
    for suffix, label in (("local", "repo_only"), ("history_only", "repo_history_only")):
        baseline_key=_part_key(f"{checkpoint_unit}_{suffix}", finish_buffer_hours)
        baseline=pd.read_parquet(PARTS / f"{baseline_key}.parquet")
        point_adapter=metric_row(adapter_pred.label,adapter_pred.probability,.5,"x","x","x")
        point_baseline=metric_row(baseline.label,baseline.probability,.5,"x","x","x")
        prefix=f"paired_weekly_adapter_minus_{label}_"
        for metric in ("roc_auc", "recall_at_15", "lift_at_15"):
            row[f"{prefix}{metric}"]=point_adapter[metric]-point_baseline[metric]
        for key, value in paired_weekly_delta_ci(baseline,adapter_pred,.5).items():
            row[key.replace("paired_weekly_delta_",prefix)]=value
    adapter_json.write_text(json.dumps([row],indent=2),encoding="utf-8")


def _assert_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    assert train.gh_build_started_at.max() < test.gh_build_started_at.min(), "train does not strictly precede test"
    assert not set(train.tr_build_id).intersection(test.tr_build_id), "build_id appears on both sides"


def _assert_history(data: pd.DataFrame) -> None:
    """Independently verify reconstructed all-prior history, including values."""
    for repo, d in data.groupby("gh_project_name", sort=False):
        pending=[]; completed=[]
        ordered=d.sort_values(["gh_build_started_at", "tr_build_id"], kind="mergesort")
        completed_count=failures=streak=0; last_finish=last_failure=None
        fields=["gh_build_started_at", "_finished_at", "tr_build_id", "target_failed", *HISTORY, *HISTORY_PLUS]
        for values in ordered[fields].itertuples(index=False, name=None):
            start, finish, build_id, target, prev, rate5, rateall, rate10, rate20, got_streak, since_build, since_failure = values
            while pending and pending[0][0] < start.value:
                finished, _start_ns, _build_id, label, _row_index=heapq.heappop(pending)
                completed.append(label); completed_count += 1; failures += label
                streak = streak + 1 if label else 0
                last_finish=finished
                if label: last_failure=finished
            if completed:
                assert prev == completed[-1], f"unsafe previous outcome for {repo}"
                assert np.isclose(rateall, failures/completed_count), f"unsafe all-history rate for {repo}"
                assert np.isclose(rate5, np.mean(completed[-5:])), f"unsafe 5-build rate for {repo}"
                assert np.isclose(rate10, np.mean(completed[-10:])), f"unsafe 10-build rate for {repo}"
                assert np.isclose(rate20, np.mean(completed[-20:])), f"unsafe 20-build rate for {repo}"
                assert got_streak == streak, f"unsafe failure streak for {repo}"
                assert np.isclose(since_build, (start.value-last_finish)/3.6e12), f"unsafe time-since-build for {repo}"
                if last_failure is None: assert pd.isna(since_failure), f"unsafe missing failure history for {repo}"
                else: assert np.isclose(since_failure, (start.value-last_failure)/3.6e12), f"unsafe time-since-failure for {repo}"
            else:
                assert all(pd.isna(value) for value in (prev,rate5,rateall,rate10,rate20,got_streak,since_build,since_failure)), f"history exists before a completed build for {repo}"
            heapq.heappush(pending, (finish.value, start.value, int(build_id), int(target), 0))


def run_travis_unit(unit: str, smoke: bool = False, finish_buffer_hours: int = 0) -> None:
    """Fit one checkpointable Travis unit; CIs are computed from saved predictions only."""
    if _load_part(unit, finish_buffer_hours):
        print(f"SKIP {unit}: checkpoint exists"); return
    started = time.perf_counter()
    data, metadata = load_travis_builds(finish_buffer_hours)
    if smoke: data = data.iloc[:2_000].copy()
    data = add_safe_history(data)
    n=len(data); train_end=int(n*.72); validation_end=int(n*.80)
    train, val, test = data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]
    _assert_split(train, test); _assert_history(data)
    pieces=unit.split("_")
    if len(pieces) != 4 or pieces[:2] != ["travis", "final"]:
        raise ValueError("Travis units are travis_final_{A|B|C}_{majority|logistic|rf|rfcal|repo_prior}")
    tier, kind=pieces[2], pieces[3]; numeric,categorical=tier_features(tier)
    if kind == "rfcal":
        _model,_calibrator,choice,threshold,p,val_brier,val_p=calibrate(train,val,test,numeric,categorical)
        name="rf_calibrated"
    elif kind == "repo_prior":
        p=repo_prior_scores(train,test); threshold=float(train.target_failed.mean()); name="repo_prior"; choice=None; val_brier=np.nan; val_p=np.nan
    else:
        model=pipeline(kind,numeric,categorical).fit(train[numeric+categorical],train.target_failed)
        p=model.predict_proba(test[numeric+categorical])[:,1]; threshold=.5; name=kind; choice=None; val_brier=np.nan; val_p=np.nan
    pred=pd.DataFrame({"repo":test.gh_project_name,"build_id":test.tr_build_id,"trigger_time":test.gh_build_started_at,
                       "label":test.target_failed,"y":test.target_failed,"probability":p,"p":p,
                       "fold_window":"final_20pct_development_test"})
    row=metric_row(pred.y,pred.p,threshold,name,tier,"final_20pct_development_test")
    macro_auc, macro_auc_sd, macro_n = _within_repo_auc(pred)
    row.update({"macro_within_repo_auc":macro_auc,"macro_within_repo_auc_sd":macro_auc_sd,"within_repo_auc_repositories":macro_n})
    row.update({"job_rows":metadata["job_rows"],"build_count":metadata["builds"],"finish_buffer_hours":finish_buffer_hours,"calibrator":choice,
                "validation_brier":val_brier,"validation_constant_brier":val_p*(1-val_p) if pd.notna(val_p) else np.nan,
                "validation_brier_skill":1-val_brier/(val_p*(1-val_p)) if pd.notna(val_p) and val_p not in (0,1) else np.nan})
    elapsed=time.perf_counter()-started
    _save_part(unit,row,pred,elapsed,finish_buffer_hours); print(f"DONE {_part_key(unit, finish_buffer_hours)} {elapsed:.1f}s")


def run_gha_unit(unit: str) -> None:
    """One checkpointable GHA temporal or eligible-repository holdout unit."""
    if _load_part(unit, 0): print(f"SKIP {unit}: checkpoint exists"); return
    pieces=unit.split("_")
    if len(pieces) < 4 or pieces[0] != "gha":
        raise ValueError("GHA units: gha_final_{first|all}_{majority|logistic|rf|repo_prior}; gha_loro_{first|all}_{repo_slug}")
    started=time.perf_counter(); attempt=pieces[2]
    if attempt not in {"first","all"}: raise ValueError("attempt must be first or all")
    data=load_gha(first_attempt_only=attempt == "first"); cut=int(len(data)*.8); train,test=data.iloc[:cut],data.iloc[cut:]
    if pieces[1] == "final":
        kind="_".join(pieces[3:])
        if kind == "repo_prior": p=repo_prior_scores(train.rename(columns={"repo":"gh_project_name"}),test.rename(columns={"repo":"gh_project_name"})); threshold=float(train.target_failed.mean()); name=kind
        else:
            model=pipeline(kind,GHA_NUM,GHA_CAT).fit(train[GHA_NUM+GHA_CAT],train.target_failed)
            p=model.predict_proba(test[GHA_NUM+GHA_CAT])[:,1]; threshold=.5; name=kind
        split=f"gha_{attempt}_attempt_temporal_final_20pct"; held=None
    elif pieces[1] == "loro":
        slug="_".join(pieces[3:]); mapping={repo.replace("/","--"):repo for repo in test.repo.unique()}
        if slug not in mapping: raise ValueError(f"unknown GHA repository slug {slug}")
        held=mapping[slug]; eligible=test.groupby("repo").target_failed.agg(builds="size",failures="sum")
        if held not in eligible.query("builds >= 200 and failures >= 15").index: raise ValueError(f"{held} does not satisfy eligibility")
        tr=train.loc[train.repo.ne(held)]; te=test.loc[test.repo.eq(held)]
        model=pipeline("rf",GHA_NUM,GHA_CAT).fit(tr[GHA_NUM+GHA_CAT],tr.target_failed); p=model.predict_proba(te[GHA_NUM+GHA_CAT])[:,1]
        test=te; threshold=.5; name="rf"; split=f"gha_loro_{attempt}_attempt" 
    else: raise ValueError("GHA unit second component must be final or loro")
    pred=pd.DataFrame({"repo":test.repo,"build_id":test.run_id,"trigger_time":test.created_at,"label":test.target_failed,"y":test.target_failed,"probability":p,"p":p,"fold_window":split})
    row=metric_row(pred.label,pred.probability,threshold,name,"GHA",split); row["dataset"]="gha"; row["attempt_policy"]=attempt
    macro,sd,nrepo=_within_repo_auc(pred); row.update({"macro_within_repo_auc":macro,"macro_within_repo_auc_sd":sd,"within_repo_auc_repositories":nrepo})
    if held is not None: row.update({"repository":held,"eligibility_rule":"later 20%: >=200 runs and >=15 failures"})
    _save_part(unit,row,pred,time.perf_counter()-started,0); print(f"DONE {_part_key(unit,0)} {row.get('elapsed_seconds','saved')}")


_WALK_CACHE: dict[int, tuple[pd.DataFrame, dict]] = {}


def _walk_data(finish_buffer_hours: int) -> tuple[pd.DataFrame, dict]:
    """Load, reconstruct and verify history once per process and buffer."""
    if finish_buffer_hours not in _WALK_CACHE:
        data,meta=load_travis_builds(finish_buffer_hours); data=add_stream_history(add_safe_history(data))
        _assert_history(data); _assert_stream_history(data)
        _WALK_CACHE[finish_buffer_hours]=(data,meta)
    return _WALK_CACHE[finish_buffer_hours]


def run_travis_walk_unit(unit: str, finish_buffer_hours: int) -> None:
    """Expanding global chronology: fold j tests after folds 1..j-1 only."""
    if _load_part(unit, finish_buffer_hours): print(f"SKIP {unit}: checkpoint exists"); return
    pieces=unit.split("_")
    if len(pieces) < 5 or pieces[:2] != ["travis","walk"]:
        raise ValueError("walk units: travis_walk_{6|7|8|9|10}_{A|B|C|Cplus|Dcommit|Dcontext|Dnorm|Dall}_{rf|balanced_rf|repo_prior|historyonly}")
    fold=int(pieces[2]); tier=pieces[3]; kind="_".join(pieces[4:])
    if fold not in range(6,11): raise ValueError("walk test fold must be 6..10")
    started=time.perf_counter(); data,meta=_walk_data(finish_buffer_hours)
    indices=np.array_split(np.arange(len(data)),10); train=data.iloc[np.concatenate(indices[:fold-1])]; test=data.iloc[indices[fold-1]]
    _assert_split(train,test)
    numeric,categorical=tier_features(tier)
    if kind == "repo_prior": p=repo_prior_scores(train,test); threshold=float(train.target_failed.mean()); name=kind
    elif kind == "historyonly":
        if tier not in {"C", "Cplus"}: raise ValueError("historyonly requires Tier C or Cplus history columns")
        # Same pre-build, local-persistence baseline used in LORO: no model
        # fit and no current/future outcome.  The training prevalence only
        # resolves the first-build / no-completed-history case.
        p=test.hist_repo_fail_rate_5.fillna(float(train.target_failed.mean())).to_numpy(float)
        threshold=float(train.target_failed.mean()); name="history_only_last5"
    elif kind in {"lgbm","lgbmrank","lgbmblend"}:
        p=fit_predict_lgbm(kind,train,test,numeric,categorical); threshold=.5; name=kind
    else:
        if kind not in {"majority","logistic","rf","balanced_rf"}: raise ValueError(f"unsupported walk model: {kind}")
        model=fit_model(kind,train,numeric,categorical)
        p=model.predict_proba(test[numeric+categorical])[:,1]; threshold=.5; name=kind
    split=f"walk_forward_fold_{fold}"
    pred=pd.DataFrame({"repo":test.gh_project_name,"build_id":test.tr_build_id,"trigger_time":test.gh_build_started_at,"label":test.target_failed,"y":test.target_failed,"probability":p,"p":p,"fold_window":split})
    row=metric_row(pred.label,pred.probability,threshold,name,tier,split); macro,sd,nrepo=_within_repo_auc(pred)
    row.update({"job_rows":meta["job_rows"],"build_count":meta["builds"],"finish_buffer_hours":finish_buffer_hours,"macro_within_repo_auc":macro,"macro_within_repo_auc_sd":sd,"within_repo_auc_repositories":nrepo})
    _save_part(unit,row,pred,time.perf_counter()-started,finish_buffer_hours); print(f"DONE {_part_key(unit,finish_buffer_hours)}")


def walk_tier_c_shuffle_control(fold: int = 8, seeds: int = 10, finish_buffer_hours: int = 0) -> None:
    """Fixed, additional walk-forward null control; it does not select models.

    Global train-label permutation is deliberately used here.  It preserves
    the feature distribution and test labels while severing feature-outcome
    association in the training window.  The result is compared as a null
    distribution, never against a target score.
    """
    if fold not in range(6, 11): raise ValueError("shuffle fold must be 6..10")
    path=REPORTS / f"phase1_v2_walk_fold_{fold}_tier_c_global_shuffle_null.csv"
    if path.exists():
        print(f"SKIP walk Tier C shuffle: {path.name} exists"); return
    data,_=load_travis_builds(finish_buffer_hours); data=add_safe_history(data)
    indices=np.array_split(np.arange(len(data)),10); train=data.iloc[np.concatenate(indices[:fold-1])].copy(); test=data.iloc[indices[fold-1]].copy()
    _assert_split(train,test); _assert_history(pd.concat([train,test]))
    numeric,categorical=tier_features("C"); records=[]
    for offset in range(seeds):
        shuffled=train.copy(); shuffled["target_failed"]=np.random.default_rng(SEED+offset).permutation(shuffled.target_failed.to_numpy())
        model=pipeline("rf",numeric,categorical).fit(shuffled[numeric+categorical],shuffled.target_failed)
        p=model.predict_proba(test[numeric+categorical])[:,1]
        pred=pd.DataFrame({"repo":test.gh_project_name,"label":test.target_failed,"probability":p})
        macro,sd,nrepo=_within_repo_auc(pred)
        records.append({"fold":fold,"seed":SEED+offset,"tier":"C","control":"global_train_label_shuffle","pooled_auc":roc_auc_score(pred.label,pred.probability),"macro_within_repo_auc":macro,"macro_within_repo_auc_sd":sd,"eligible_repositories":nrepo,"finish_buffer_hours":finish_buffer_hours})
    draws=pd.DataFrame(records); draws.to_csv(path,index=False)
    summary=draws.agg({"pooled_auc":["mean","std",lambda x:x.quantile(.025),lambda x:x.quantile(.975)],"macro_within_repo_auc":["mean","std",lambda x:x.quantile(.025),lambda x:x.quantile(.975)]})
    summary.to_csv(REPORTS / f"phase1_v2_walk_fold_{fold}_tier_c_global_shuffle_null_summary.csv")
    print(draws.to_string(index=False))


def random_score_control(fold: int = 8, seeds: int = 1000, finish_buffer_hours: int = 0) -> None:
    """Independent scoring sanity check using saved, fixed Tier-C test labels.

    This validates the within-repository AUC implementation.  It is not a
    training-label permutation and therefore does not diagnose the separate
    below-chance shuffled-model phenomenon.
    """
    key=_part_key(f"travis_walk_{fold}_C_rf", finish_buffer_hours)
    source=PARTS / f"{key}.parquet"
    if not source.exists(): raise ValueError(f"missing saved Tier C predictions: {source.name}")
    output=REPORTS / f"phase1_v2_walk_fold_{fold}_random_score_null.csv"
    if output.exists(): print(f"SKIP random-score control: {output.name} exists"); return
    pred=pd.read_parquet(source); records=[]
    for offset in range(seeds):
        scored=pred[["repo","label"]].copy(); scored["probability"]=np.random.default_rng(SEED+offset).random(len(scored))
        macro,sd,nrepo=_within_repo_auc(scored)
        records.append({"fold":fold,"seed":SEED+offset,"control":"independent_uniform_random_scores","macro_within_repo_auc":macro,"macro_within_repo_auc_sd":sd,"eligible_repositories":nrepo,"finish_buffer_hours":finish_buffer_hours})
    draws=pd.DataFrame(records); draws.to_csv(output,index=False)
    summary=pd.DataFrame([{"fold":fold,"draws":seeds,"macro_within_repo_auc_mean":draws.macro_within_repo_auc.mean(),"macro_within_repo_auc_sd":draws.macro_within_repo_auc.std(ddof=1),"macro_within_repo_auc_low":draws.macro_within_repo_auc.quantile(.025),"macro_within_repo_auc_high":draws.macro_within_repo_auc.quantile(.975),"contains_0_5":bool(draws.macro_within_repo_auc.quantile(.025)<=.5<=draws.macro_within_repo_auc.quantile(.975))}])
    summary.to_csv(REPORTS / f"phase1_v2_walk_fold_{fold}_random_score_null_summary.csv",index=False)
    print(summary.to_string(index=False))


def run_travis_project_unit(unit: str, finish_buffer_hours: int) -> None:
    """One repository/fold of the predeclared 10-fold online evaluation."""
    if _load_part(unit, finish_buffer_hours): print(f"SKIP {unit}: checkpoint exists"); return
    # Slash is represented by -- so the part filename remains portable.
    prefix="travis_project_"; marker="_fold_"
    if not unit.startswith(prefix) or marker not in unit: raise ValueError("project unit: travis_project_{repo--name}_fold_{6..10}_{A|C}_rf_ros")
    slug,tail=unit[len(prefix):].split(marker,1); fold_text,tier,*kind_parts=tail.split("_"); kind="_".join(kind_parts)
    fold=int(fold_text); repo=slug.replace("--","/")
    if fold not in range(6,11) or kind != "rf_ros": raise ValueError("project units support folds 6..10 and rf_ros")
    started=time.perf_counter(); data,meta=load_travis_builds(finish_buffer_hours); data=add_safe_history(data)
    d=data.loc[data.gh_project_name.eq(repo)].sort_values(["gh_build_started_at","tr_build_id"],kind="mergesort").reset_index(drop=True)
    if len(d)==0: raise ValueError(f"repository not found: {repo}")
    folds=np.array_split(np.arange(len(d)),10); train=d.iloc[np.concatenate(folds[:fold-1])]; test=d.iloc[folds[fold-1]]
    if train.target_failed.sum()==0 or test.target_failed.sum()==0: raise ValueError("skip: no failures in train or test fold")
    _assert_split(train,test); numeric,categorical=tier_features(tier)
    preprocess,model,sampling=fit_project_resampled_rf(train,numeric,categorical); p=model.predict_proba(preprocess.transform(test[numeric+categorical]))[:,1]
    split=f"per_project_online_fold_{fold}"; pred=pd.DataFrame({"repo":test.gh_project_name,"build_id":test.tr_build_id,"trigger_time":test.gh_build_started_at,"label":test.target_failed,"y":test.target_failed,"probability":p,"p":p,"fold_window":split})
    row=metric_row(pred.label,pred.probability,.5,"rf_ros",tier,split); row.update({"repository":repo,"job_rows":meta["job_rows"],"build_count":meta["builds"],"finish_buffer_hours":finish_buffer_hours,**sampling})
    _save_part(unit,row,pred,time.perf_counter()-started,finish_buffer_hours); print(f"DONE {_part_key(unit,finish_buffer_hours)}")


def run_travis_project_online_repository(unit: str, finish_buffer_hours: int) -> None:
    """All five online folds for one repository, loading Travis once."""
    prefix="travis_project_online_"; suffix="_C_rf_ros"
    if not unit.startswith(prefix) or not unit.endswith(suffix):
        raise ValueError("project online unit: travis_project_online_{repo--name}_C_rf_ros")
    slug=unit[len(prefix):-len(suffix)]; repo=slug.replace("--","/")
    part_units=[f"travis_project_{slug}_fold_{fold}_C_rf_ros" for fold in range(6,11)]
    if all(_load_part(part,finish_buffer_hours) for part in part_units):
        print(f"SKIP {unit}: all five fold checkpoints exist"); return
    started=time.perf_counter(); data,meta=load_travis_builds(finish_buffer_hours); data=add_safe_history(data)
    d=data.loc[data.gh_project_name.eq(repo)].sort_values(["gh_build_started_at","tr_build_id"],kind="mergesort").reset_index(drop=True)
    if len(d)==0: raise ValueError(f"repository not found: {repo}")
    folds=np.array_split(np.arange(len(d)),10); numeric,categorical=tier_features("C")
    for fold in range(6,11):
        part=part_units[fold-6]
        if _load_part(part,finish_buffer_hours): print(f"SKIP {part}: checkpoint exists"); continue
        train=d.iloc[np.concatenate(folds[:fold-1])]; test=d.iloc[folds[fold-1]]
        if train.target_failed.sum()==0 or test.target_failed.sum()==0:
            _record_skip(part,"per-project online fold has no failures in train or test",finish_buffer_hours); print(f"SKIP {part}: no failures in train or test"); continue
        _assert_split(train,test)
        preprocess,model,sampling=fit_project_resampled_rf(train,numeric,categorical)
        p=model.predict_proba(preprocess.transform(test[numeric+categorical]))[:,1]
        split=f"per_project_online_fold_{fold}"
        pred=pd.DataFrame({"repo":test.gh_project_name,"build_id":test.tr_build_id,"trigger_time":test.gh_build_started_at,"label":test.target_failed,"y":test.target_failed,"probability":p,"p":p,"fold_window":split})
        row=metric_row(pred.label,pred.probability,.5,"rf_ros","C",split)
        row.update({"repository":repo,"job_rows":meta["job_rows"],"build_count":meta["builds"],"finish_buffer_hours":finish_buffer_hours,**sampling})
        _save_part(part,row,pred,time.perf_counter()-started,finish_buffer_hours); print(f"DONE {_part_key(part,finish_buffer_hours)}")


def run_travis_loro_unit(unit: str, finish_buffer_hours: int) -> None:
    """Temporal leave-one-repository-out on the three predeclared Travis repos."""
    prefix="travis_loro_"
    if not unit.startswith(prefix): raise ValueError("LORO unit: travis_loro_{repo--name}_{A|B|C}_rf")
    slug,tier,kind=unit[len(prefix):].rsplit("_",2); repo=slug.replace("--","/")
    if kind != "rf": raise ValueError("LORO currently supports rf")
    # Earlier LORO parts predate either the strict finish-time purge or the
    # explicit finish/start/build-id tie order for history. Keep them intact,
    # but checkpoint this fully specified protocol separately.
    checkpoint_unit=f"{unit}_refined_finish_purge_tie_order_v2"
    required_parts=[checkpoint_unit, *(f"{checkpoint_unit}_{suffix}" for suffix in ("local", "calibrated", "adapter", "history_only", "always_alert", "majority"))]
    if all(_load_part(part, finish_buffer_hours) for part in required_parts):
        _add_loro_adapter_comparisons(checkpoint_unit, finish_buffer_hours)
        print(f"SKIP {checkpoint_unit}: all revised checkpoints exist; refreshed paired comparisons from saved predictions"); return
    started=time.perf_counter(); data,meta=load_travis_builds(finish_buffer_hours); data=add_safe_history(data)
    target=data.loc[data.gh_project_name.eq(repo)].sort_values(["gh_build_started_at","tr_build_id"],kind="mergesort").reset_index(drop=True)
    if len(target)==0: raise ValueError(f"repository not found: {repo}")
    # Predeclared repository-local final 20%, with all target rows absent from train.
    test_start=target.iloc[int(.8*len(target))].gh_build_started_at
    test=target.loc[target.gh_build_started_at.ge(test_start)].copy()
    # Strict temporal purge: training outcomes must have completed before the
    # held-out repository's final-test period starts, not merely started before.
    train=data.loc[data.gh_project_name.ne(repo) & data._finished_at.lt(test_start)].copy()
    _assert_split(train,test); assert not train.gh_project_name.eq(repo).any(), "held-out repository in train"
    _assert_history(data)
    numeric,categorical=tier_features(tier); model=pipeline("rf",numeric,categorical).fit(train[numeric+categorical],train.target_failed)
    split="temporal_leave_one_repository_out"; p=model.predict_proba(test[numeric+categorical])[:,1]
    pred=pd.DataFrame({"repo":test.gh_project_name,"build_id":test.tr_build_id,"trigger_time":test.gh_build_started_at,"label":test.target_failed,"y":test.target_failed,"probability":p,"p":p,"fold_window":split,"variant":"global_cold_start"})
    row=metric_row(pred.label,pred.probability,.5,"rf",tier,split); row.update(weekly_block_ci(pred,.5)); row.update({"repository":repo,"variant":"global_cold_start","train_rows":len(train),"test_rows":len(test),"test_start":str(test_start),"job_rows":meta["job_rows"],"build_count":meta["builds"],"finish_buffer_hours":finish_buffer_hours})
    row.update({"max_recall_at_15_cap":min(1.0,.15/float(test.target_failed.mean())),"max_lift_at_15_cap":1/float(test.target_failed.mean())})
    _save_part(checkpoint_unit,row,pred,time.perf_counter()-started,finish_buffer_hours)
    # Same final 20%: a local model and a calibrated global score are adaptation
    # comparisons, not the cold-start primary estimate.
    local_train=target.loc[target.gh_build_started_at.lt(test_start)].copy()
    local=pipeline("rf",numeric,categorical).fit(local_train[numeric+categorical],local_train.target_failed)
    local_p=local.predict_proba(test[numeric+categorical])[:,1]
    global_early_p=model.predict_proba(local_train[numeric+categorical])[:,1]
    calibrator=ProbabilityCalibrator.fit("sigmoid",global_early_p,local_train.target_failed)
    calibrated_p=calibrator.transform(p)
    adapter_p=fit_repo_ranking_adapter(global_early_p,local_train,p,test)
    for suffix, scores, name in (("local",local_p,"repo_only"),("calibrated",calibrated_p,"global_plus_repo_sigmoid_calibration_only"),("adapter",adapter_p,"global_plus_repo_history_adapter")):
        part=f"{checkpoint_unit}_{suffix}"; prediction=pred.copy(); prediction["probability"]=scores; prediction["p"]=scores; prediction["variant"]=name
        result=metric_row(prediction.label,prediction.probability,.5,name,tier,split); result.update(weekly_block_ci(prediction,.5)); result.update(paired_weekly_delta_ci(pred,prediction,.5)); result.update({"repository":repo,"variant":name,"train_rows":len(local_train) if suffix=="local" else len(train),"adaptation_rows":len(local_train),"test_rows":len(test),"test_start":str(test_start),"job_rows":meta["job_rows"],"build_count":meta["builds"],"finish_buffer_hours":finish_buffer_hours})
        result.update({"max_recall_at_15_cap":min(1.0,.15/float(test.target_failed.mean())),"max_lift_at_15_cap":1/float(test.target_failed.mean()),"calibration_only_monotonic":suffix=="calibrated"})
        _save_part(part,result,prediction,time.perf_counter()-started,finish_buffer_hours)
    # Pure local history baseline: no global model and no current test outcome.
    history_scores=test.hist_repo_fail_rate_5.fillna(float(local_train.target_failed.mean())).to_numpy(float)
    history_prediction=pred.copy(); history_prediction["probability"]=history_scores; history_prediction["p"]=history_scores; history_prediction["variant"]="repo_history_only"
    history_result=metric_row(history_prediction.label,history_prediction.probability,.5,"repo_history_only",tier,split); history_result.update(weekly_block_ci(history_prediction,.5)); history_result.update(paired_weekly_delta_ci(pred,history_prediction,.5)); history_result.update({"repository":repo,"variant":"repo_history_only","adaptation_rows":len(local_train),"test_rows":len(test),"finish_buffer_hours":finish_buffer_hours,"max_recall_at_15_cap":min(1.0,.15/float(test.target_failed.mean())),"max_lift_at_15_cap":1/float(test.target_failed.mean())})
    _save_part(f"{checkpoint_unit}_history_only",history_result,history_prediction,time.perf_counter()-started,finish_buffer_hours)
    # Reference decision baselines, especially material for dd-agent's >50% prevalence.
    for suffix, scores, threshold, name in (("always_alert",np.ones(len(test)),.5,"always_alert"),("majority",np.full(len(test),int(local_train.target_failed.mean() >= .5)),.5,"majority_class")):
        part=f"{checkpoint_unit}_{suffix}"; prediction=pred.copy(); prediction["probability"]=scores; prediction["p"]=scores; prediction["variant"]=name
        result=metric_row(prediction.label,prediction.probability,threshold,name,tier,split)
        for key in ("precision_at_10","recall_at_10","lift_at_10","precision_at_15","recall_at_15","lift_at_15"):
            result[key]=np.nan
        result.update({"repository":repo,"variant":name,"top_k_metric_note":"undefined for tied constant scores; use threshold-based metrics","adaptation_rows":len(local_train),"test_rows":len(test),"finish_buffer_hours":finish_buffer_hours,"max_recall_at_15_cap":min(1.0,.15/float(test.target_failed.mean())),"max_lift_at_15_cap":1/float(test.target_failed.mean())})
        _save_part(part,result,prediction,time.perf_counter()-started,finish_buffer_hours)
    _add_loro_adapter_comparisons(checkpoint_unit, finish_buffer_hours)
    print(f"DONE {_part_key(checkpoint_unit,finish_buffer_hours)} with local, calibration, adapter, history-only, and decision baselines")


def smoke() -> None:
    """Fast structural test; it never supplies reportable model results."""
    started=time.perf_counter(); data,_=load_travis_builds(); data=add_safe_history(data.iloc[:20_000].copy())
    train,test=data.iloc[:16_000],data.iloc[16_000:]; _assert_split(train,test); _assert_history(data)
    # Negative control: an explicitly leaky history must be rejected by the
    # independent value-level history assertion.
    leaky=data.copy(); leaky["hist_repo_fail_rate_all"]=leaky.target_failed.astype(float)
    timing_leak_caught=False
    try: _assert_history(leaky)
    except AssertionError: timing_leak_caught=True
    assert timing_leak_caught, "deliberate current-outcome history leak was not detected"
    full,_=load_travis_builds(); full=add_stream_history(add_safe_history(full))
    _assert_stream_history(full)
    leaky_stream=full.copy(); leaky_stream["hist_stream_prev_failed"]=leaky_stream.target_failed.astype(float)
    stream_leak_caught=False
    try: _assert_stream_history(leaky_stream)
    except AssertionError: stream_leak_caught=True
    assert stream_leak_caught, "deliberate current-outcome stream leak was not detected"
    aucs={}
    for tier in ("A","B","C"):
        numeric,categorical=tier_features(tier)
        shuffled=train.copy(); shuffled["target_failed"]=np.random.default_rng(SEED).permutation(shuffled.target_failed.to_numpy())
        p=pipeline("rf",numeric,categorical).fit(shuffled[numeric+categorical],shuffled.target_failed).predict_proba(test[numeric+categorical])[:,1]
        aucs[tier]=roc_auc_score(test.target_failed,p)
    print(json.dumps({"smoke": "passed", "travis_builds":len(data), "train":len(train), "test":len(test), "timing_leak_caught":timing_leak_caught, "stream_leak_caught":stream_leak_caught, "label_shuffle_auc":aucs, "elapsed_seconds":time.perf_counter()-started},indent=2))


def _within_repo_auc(pred: pd.DataFrame) -> tuple[float, float, int]:
    values=[roc_auc_score(d.label,d.probability) for _,d in pred.groupby("repo") if d.label.nunique()==2]
    return float(np.mean(values)), float(np.std(values,ddof=1)) if len(values)>1 else np.nan, len(values)


def repo_prior_scores(train: pd.DataFrame, test: pd.DataFrame, repo_column: str = "gh_project_name") -> np.ndarray:
    """Repository-rate baseline; unseen repositories receive the global training rate."""
    return test[repo_column].map(train.groupby(repo_column).target_failed.mean()).fillna(float(train.target_failed.mean())).to_numpy(float)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p=np.clip(np.asarray(probabilities,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))


def fit_repo_ranking_adapter(global_train_scores: np.ndarray, local_train: pd.DataFrame, global_test_scores: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    """Local rank adaptation: global logit + pre-build local history, no test labels."""
    columns=["global_score_logit", "hist_repo_fail_rate_5", "hist_prev_failed"]
    x_train=pd.DataFrame({"global_score_logit":_logit(global_train_scores), "hist_repo_fail_rate_5":local_train.hist_repo_fail_rate_5, "hist_prev_failed":local_train.hist_prev_failed})
    x_test=pd.DataFrame({"global_score_logit":_logit(global_test_scores), "hist_repo_fail_rate_5":test.hist_repo_fail_rate_5, "hist_prev_failed":test.hist_prev_failed})
    adapter=Pipeline([("impute",SimpleImputer(strategy="median")),("model",LogisticRegression(max_iter=500,class_weight="balanced",random_state=SEED))])
    adapter.fit(x_train[columns],local_train.target_failed)
    return adapter.predict_proba(x_test[columns])[:,1]


def shuffle_controls() -> None:
    """20-seed global/within-repository label-shuffle null distributions.

    No individual shuffled AUC is required to fall in a fixed interval.  The
    predeclared pass criterion is that the 95% interval for *macro within-repo*
    AUC under global shuffle contains 0.5.  Pooled AUC is retained as a null
    distribution for empirical comparisons, not a pass/fail test.
    """
    data,_=load_travis_builds(); data=add_safe_history(data.iloc[:20_000].copy())
    train,test=data.iloc[:16_000],data.iloc[16_000:]
    records=[]
    for tier in ("A","B","C"):
        numeric,categorical=tier_features(tier)
        for seed_offset in range(20):
            rng=np.random.default_rng(SEED+seed_offset)
            for control in ("global_train_label_shuffle", "within_repo_train_label_shuffle"):
                shuffled=train.copy()
                if control.startswith("global"):
                    shuffled["target_failed"]=rng.permutation(shuffled.target_failed.to_numpy())
                else:
                    shuffled["target_failed"]=shuffled.groupby("gh_project_name", group_keys=False).target_failed.transform(lambda s: rng.permutation(s.to_numpy()))
                p=pipeline("rf",numeric,categorical).fit(shuffled[numeric+categorical],shuffled.target_failed).predict_proba(test[numeric+categorical])[:,1]
                pred=pd.DataFrame({"repo":test.gh_project_name,"label":test.target_failed,"probability":p})
                macro, macro_sd, n_repos=_within_repo_auc(pred)
                records.append({"tier":tier,"control":control,"seed":SEED+seed_offset,"pooled_auc":roc_auc_score(pred.label,pred.probability),"macro_within_repo_auc":macro,"macro_within_repo_auc_sd":macro_sd,"eligible_test_repositories":n_repos})
    d=pd.DataFrame(records); summary=(d.groupby(["tier","control"])
        .agg(pooled_auc_mean=("pooled_auc","mean"),pooled_auc_sd=("pooled_auc","std"),pooled_auc_low=("pooled_auc",lambda x:x.quantile(.025)),pooled_auc_high=("pooled_auc",lambda x:x.quantile(.975)),
             macro_within_repo_auc_mean=("macro_within_repo_auc","mean"),macro_within_repo_auc_sd=("macro_within_repo_auc","std"),macro_within_repo_auc_low=("macro_within_repo_auc",lambda x:x.quantile(.025)),macro_within_repo_auc_high=("macro_within_repo_auc",lambda x:x.quantile(.975)),eligible_test_repositories=("eligible_test_repositories","first")).reset_index())
    global_summary=summary.loc[summary.control.eq("global_train_label_shuffle")]
    assert ((global_summary.macro_within_repo_auc_low <= .5) & (global_summary.macro_within_repo_auc_high >= .5)).all(), "global-shuffle macro within-repo null interval does not contain 0.5"
    d.to_csv(REPORTS/"phase1_v2_shuffle_null_draws.csv",index=False); summary.to_csv(REPORTS/"phase1_v2_shuffle_null_summary.csv",index=False)
    print(summary.to_string(index=False))


def rf_shuffle_investigation() -> None:
    """Diagnostic only: fixed A-tier global shuffle across RF size settings."""
    data,_=load_travis_builds(); data=add_safe_history(data.iloc[:20_000].copy()); train,test=data.iloc[:16_000],data.iloc[16_000:]
    numeric,categorical=tier_features("A"); shuffled=train.copy(); shuffled["target_failed"]=np.random.default_rng(SEED).permutation(shuffled.target_failed.to_numpy())
    rows=[]
    for trees, leaf in ((200,2),(200,50),(50,2),(500,2)):
        model=pipeline("rf",numeric,categorical); model.named_steps["model"].set_params(n_estimators=trees,min_samples_leaf=leaf)
        model.fit(shuffled[numeric+categorical],shuffled.target_failed); p=model.predict_proba(test[numeric+categorical])[:,1]
        pred=pd.DataFrame({"repo":test.gh_project_name,"label":test.target_failed,"probability":p})
        macro,macro_sd,nrepo=_within_repo_auc(pred)
        # Score concentration checks whether extreme/sparse-region scores drive the effect.
        rows.append({"trees":trees,"min_samples_leaf":leaf,"pooled_auc":roc_auc_score(test.target_failed,p),"macro_within_repo_auc":macro,"macro_within_repo_auc_sd":macro_sd,"score_p01":np.quantile(p,.01),"score_p99":np.quantile(p,.99),"score_unique":len(np.unique(p)),"eligible_repositories":nrepo})
    pd.DataFrame(rows).to_csv(REPORTS/"phase1_v2_rf_shuffle_investigation.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser=argparse.ArgumentParser(description="Resumable Phase-1 build-risk evaluation")
    parser.add_argument("--units", default="", help="Comma-separated unit IDs; existing parts are skipped.")
    parser.add_argument("--smoke", action="store_true", help="Run only fast chronology/leakage assertions.")
    parser.add_argument("--shuffle-controls", action="store_true")
    parser.add_argument("--rf-shuffle-investigation", action="store_true")
    parser.add_argument("--walk-tier-c-shuffle-control", action="store_true", help="Fixed fold-8, 10-seed Tier C global train-label shuffle null.")
    parser.add_argument("--random-score-control", action="store_true", help="Independent uniform-random score sanity check on saved Tier-C test predictions.")
    parser.add_argument("--shuffle-fold", type=int, choices=range(6,11), default=8)
    parser.add_argument("--shuffle-seeds", type=int, default=10)
    parser.add_argument("--random-score-seeds", type=int, default=1000)
    parser.add_argument("--finish-buffer-hours", type=int, choices=(0,1,6), default=0)
    args=parser.parse_args(); REPORTS.mkdir(exist_ok=True); PARTS.mkdir(exist_ok=True)
    _write_manifest_with_duration_quality()
    if args.smoke:
        if args.units:
            for unit in [u.strip() for u in args.units.split(",") if u.strip()]:
                if unit.startswith("travis_final_"): run_travis_unit(unit, smoke=True, finish_buffer_hours=args.finish_buffer_hours)
                else: raise SystemExit("--smoke unit execution currently supports only travis_final units; use --smoke without --units for assertions.")
        else:
            smoke()
        _write_manifest_with_duration_quality(); return
    if args.shuffle_controls:
        shuffle_controls(); _write_manifest_with_duration_quality(); return
    if args.rf_shuffle_investigation:
        rf_shuffle_investigation(); _write_manifest_with_duration_quality(); return
    if args.walk_tier_c_shuffle_control:
        walk_tier_c_shuffle_control(args.shuffle_fold,args.shuffle_seeds,args.finish_buffer_hours); _write_manifest_with_duration_quality(); return
    if args.random_score_control:
        random_score_control(args.shuffle_fold,args.random_score_seeds,args.finish_buffer_hours); _write_manifest_with_duration_quality(); return
    if not args.units: raise SystemExit("Specify --units. Use scripts/merge_phase1_v2_parts.py after units complete.")
    for unit in [u.strip() for u in args.units.split(",") if u.strip()]:
        try:
            if unit.startswith("travis_final_"): run_travis_unit(unit, finish_buffer_hours=args.finish_buffer_hours)
            elif unit.startswith("travis_walk_"): run_travis_walk_unit(unit, finish_buffer_hours=args.finish_buffer_hours)
            elif unit.startswith("travis_project_online_"): run_travis_project_online_repository(unit, finish_buffer_hours=args.finish_buffer_hours)
            elif unit.startswith("travis_project_"): run_travis_project_unit(unit, finish_buffer_hours=args.finish_buffer_hours)
            elif unit.startswith("travis_loro_"): run_travis_loro_unit(unit, finish_buffer_hours=args.finish_buffer_hours)
            elif unit.startswith("gha_"): run_gha_unit(unit)
            else: raise ValueError(f"unknown unit prefix: {unit}")
        except (MemoryError, ValueError) as exc:
            _record_skip(unit,str(exc),args.finish_buffer_hours); print(f"SKIP {unit}: {exc}")
    _write_manifest_with_duration_quality()

if __name__ == "__main__": main()
