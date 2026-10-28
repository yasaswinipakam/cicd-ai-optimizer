"""Phase 2 finance pilot evaluation – protocol v2.0.

Extends v1.1 with two additions:
  1. LightGBM (Tier E model, matching Phase 1 approach)
  2. Finance-criticality badges as predictive features

Everything else (fold structure, leakage guard, CI bootstrap) is unchanged
from v1.1 so results are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "phase2_finance_v1_annotated_suites.parquet"
PARTS = ROOT / "reports" / "phase2_finance_v2_parts"
REPORTS = ROOT / "reports"
SEED = 20261013

STRICT = ["log_additions", "log_deletions", "log_total_changes", "log_changed_file_count", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
HISTORY = ["previous_completed_failure", "last5_completed_failure_rate", "all_prior_completed_failure_rate", "hours_since_completed", "hours_since_failure"]
BADGES = ["badge_payment_billing", "badge_ledger_accounting", "badge_auth_security", "badge_dependency_ci", "badge_any_finance_criticality"]
FEATURES_RF = STRICT + HISTORY
FEATURES_LGB = STRICT + HISTORY + BADGES


def part(unit: str, buffer: int) -> tuple[Path, Path]:
    key = f"{unit}__finish_buffer_hours_{buffer}"
    return PARTS / f"{key}.json", PARTS / f"{key}.parquet"


def top_metrics(y: np.ndarray, p: np.ndarray, budget: float) -> tuple[float, float, float]:
    k = max(1, int(np.ceil(len(y) * budget)))
    chosen = np.argsort(-p, kind="stable")[:k]
    precision = float(y[chosen].mean())
    recall = float(y[chosen].sum() / y.sum()) if y.sum() else np.nan
    return precision, recall, precision / float(y.mean()) if y.mean() else np.nan


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1); value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any(): value += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def metrics(y: np.ndarray, p: np.ndarray, model: str, split: str) -> dict:
    result = {"model": model, "split": split, "n": len(y), "failures": int(y.sum()), "prevalence": float(y.mean())}
    result["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    result["pr_auc"] = float(average_precision_score(y, p)) if y.sum() else np.nan
    result["precision"] = float(precision_score(y, p >= .5, zero_division=0)); result["recall"] = float(recall_score(y, p >= .5, zero_division=0)); result["f1"] = float(f1_score(y, p >= .5, zero_division=0))
    result["brier"] = float(brier_score_loss(y, p)); result["ece"] = ece(y, p)
    for pct in (10, 15):
        q, r, l = top_metrics(y, p, pct / 100); result[f"precision_at_{pct}"] = q; result[f"recall_at_{pct}"] = r; result[f"lift_at_{pct}"] = l
    result["recall_at_15_cap"] = float(min(1.0, .15 / y.mean())) if y.mean() else np.nan
    return result


def add_safe_history(data: pd.DataFrame, buffer_hours: int) -> pd.DataFrame:
    d = data.sort_values(["trigger_time", "head_sha"], kind="mergesort").copy()
    d["finish_boundary"] = d.finish_time + pd.Timedelta(hours=buffer_hours)
    d["previous_completed_failure"] = 0.; d["last5_completed_failure_rate"] = 0.; d["all_prior_completed_failure_rate"] = 0.
    d["hours_since_completed"] = -1.; d["hours_since_failure"] = -1.
    completed = []; pending = []; cursor = 0
    for idx, row in d.iterrows():
        pending.append((row.finish_boundary, row.trigger_time, row.head_sha, int(row.target_failure), idx))
    pending.sort(key=lambda x: (x[0], x[1], x[2]))
    for idx, row in d.iterrows():
        while cursor < len(pending) and pending[cursor][0] < row.trigger_time:
            completed.append(pending[cursor]); cursor += 1
        if not completed: continue
        failures = [v[3] for v in completed]
        d.at[idx, "previous_completed_failure"] = failures[-1]
        d.at[idx, "last5_completed_failure_rate"] = float(np.mean(failures[-5:]))
        d.at[idx, "all_prior_completed_failure_rate"] = float(np.mean(failures))
        d.at[idx, "hours_since_completed"] = max(0., (row.trigger_time - completed[-1][0]).total_seconds() / 3600)
        prior_fails = [v for v in completed if v[3] == 1]
        if prior_fails:
            d.at[idx, "hours_since_failure"] = max(0., (row.trigger_time - prior_fails[-1][0]).total_seconds() / 3600)
    return d


def prep(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy(); d["trigger_time"] = pd.to_datetime(d.trigger_time, utc=True); d["finish_time"] = pd.to_datetime(d.finish_time, utc=True)
    for source, target in [("additions", "log_additions"), ("deletions", "log_deletions"), ("total_changes", "log_total_changes"), ("changed_file_count", "log_changed_file_count")]:
        d[target] = np.log1p(d[source].astype(float))
    hour = d.trigger_time.dt.hour + d.trigger_time.dt.minute / 60; dow = d.trigger_time.dt.dayofweek
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24); d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["dow_sin"] = np.sin(2 * np.pi * dow / 7); d["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    for b in BADGES:
        d[b] = d[b].fillna(0).astype(float)
    return d.sort_values(["trigger_time", "head_sha"], kind="mergesort").reset_index(drop=True)


def rf(features: list[str]) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])


def lgbm(pos_weight: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def choose_calibration(raw_validation: np.ndarray, y_validation: np.ndarray) -> tuple[str, object | None, dict]:
    raw = np.clip(raw_validation, 1e-6, 1 - 1e-6); result = {"raw_brier": float(brier_score_loss(y_validation, raw))}
    if len(np.unique(y_validation)) < 2: return "raw", None, result
    calibrator = LogisticRegression(C=1e6, random_state=SEED, max_iter=2000).fit(np.log(raw / (1 - raw)).reshape(-1, 1), y_validation)
    calibrated = calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]; result["sigmoid_brier"] = float(brier_score_loss(y_validation, calibrated))
    return ("sigmoid", calibrator, result) if result["sigmoid_brier"] < result["raw_brier"] else ("raw", None, result)


def apply_calibration(p: np.ndarray, calibrator: object | None) -> np.ndarray:
    if calibrator is None: return p
    q = np.clip(p, 1e-6, 1 - 1e-6); return calibrator.predict_proba(np.log(q / (1 - q)).reshape(-1, 1))[:, 1]


def weekly_ci(pred: pd.DataFrame, draws: int = 1000) -> dict:
    d = pred.copy(); d["week"] = pd.to_datetime(d.trigger_time, utc=True).dt.to_period("W").astype(str)
    blocks = [x for _, x in d.groupby("week")]; rng = np.random.default_rng(SEED); rows = []
    for _ in range(draws):
        sample = pd.concat([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))], ignore_index=True)
        if sample.label.nunique() < 2: continue
        m = metrics(sample.label.to_numpy(int), sample.probability.to_numpy(float), "", " ")
        rows.append({k: m[k] for k in ("roc_auc", "recall_at_15", "precision_at_15", "lift_at_15")})
    table = pd.DataFrame(rows); return {f"{col}_weekly_ci_low": float(table[col].quantile(.025)) for col in table} | {f"{col}_weekly_ci_high": float(table[col].quantile(.975)) for col in table}


def save(unit: str, row: dict, pred: pd.DataFrame, buffer: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True); js, pq = part(unit, buffer)
    pred.to_parquet(pq, index=False); js.write_text(json.dumps([row], indent=2, default=str) + "\n")


def run_model(name: str, tier: str, model_fn, features: list[str], train: pd.DataFrame, val: pd.DataFrame | None,
              test: pd.DataFrame, buffer: int, unit_base: str) -> dict:
    unit = f"{unit_base}_{tier}_{name}"
    js, pq = part(unit, buffer)
    if js.exists() and pq.exists():
        print(f"SKIP {unit}: checkpoint exists")
        return json.loads(js.read_text())[0]

    X_tr = train[features]; y_tr = train.target_failure.to_numpy(int)
    X_te = test[features]; y_te = test.target_failure.to_numpy(int)

    if name == "lgbm":
        pos_weight = float((y_tr == 0).sum()) / max(1, float((y_tr == 1).sum()))
        m = lgbm(pos_weight)
        if val is not None and len(val) >= 10:
            X_val = val[features]; y_val = val.target_failure.to_numpy(int)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        else:
            m.fit(X_tr, y_tr)
        raw_val_p = m.predict_proba(val[features])[:, 1] if val is not None and len(val) >= 10 else None
        p_raw = m.predict_proba(X_te)[:, 1]
    else:
        pipeline = model_fn(features)
        pipeline.fit(X_tr, y_tr)
        raw_val_p = pipeline.predict_proba(val[features])[:, 1] if val is not None else None
        p_raw = pipeline.predict_proba(X_te)[:, 1]

    if val is not None and raw_val_p is not None:
        y_val = val.target_failure.to_numpy(int)
        choice, cal, cal_result = choose_calibration(raw_val_p, y_val)
        p = apply_calibration(p_raw, cal)
    else:
        choice, cal, cal_result, p = "none", None, {}, p_raw

    pred = pd.DataFrame({
        "repo": test.repository, "build_id": test.head_sha, "trigger_time": test.trigger_time,
        "label": y_te, "probability": p,
        "fold_window": unit_base, "model": name, "tier": tier, "calibration": choice,
    })
    row = metrics(y_te, p, name, unit_base)
    row.update({"repository": "apache/fineract", "finish_buffer_hours": buffer, "n_train": len(train),
                "n_test": len(test), "calibration": choice, **cal_result, **weekly_ci(pred)})
    save(unit, row, pred, buffer)
    print(f"DONE {unit}  AUC={row['roc_auc']:.3f}  lift@15%={row['lift_at_15']:.3f}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finish-buffer-hours", type=int, choices=(0, 1, 6), default=0)
    parser.add_argument("--units", default="final,online")
    parser.add_argument("--models", default="rf,lgbm", help="comma-separated: rf, lgbm")
    args = parser.parse_args()

    base = prep(pd.read_parquet(DATA))
    d = add_safe_history(base, args.finish_buffer_hours)
    n = len(d); units = set(args.units.split(",")); models = set(args.models.split(","))

    manifest = {
        "study": "phase2_finance_pilot_v2_0", "seed": SEED,
        "features_rf": FEATURES_RF, "features_lgbm": FEATURES_LGB,
        "badge_features_added": BADGES,
        "finish_buffer_hours": args.finish_buffer_hours,
        "rows": n, "python": sys.version, "sklearn": sklearn.__version__,
        "lightgbm": lgb.__version__, "platform": platform.platform(),
    }
    (REPORTS / "phase2_finance_v2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if "final" in units:
        cut_test = int(.8 * n); cut_val = int(.8 * cut_test)
        val_start = d.iloc[cut_val].trigger_time; test_start = d.iloc[cut_test].trigger_time
        train = d.iloc[:cut_val].loc[lambda x: x.finish_boundary < val_start]
        val = d.iloc[cut_val:cut_test].loc[lambda x: x.finish_boundary < test_start]
        test = d.iloc[cut_test:].copy()
        if train.target_failure.nunique() < 2 or val.target_failure.nunique() < 2 or test.target_failure.nunique() < 2:
            raise ValueError("final split lacks both labels")
        base_unit = "finance_fineract_final"
        if "rf" in models:
            run_model("rf", "C", lambda feats: rf(feats), FEATURES_RF, train, val, test, args.finish_buffer_hours, base_unit)
        if "lgbm" in models:
            run_model("lgbm", "E", None, FEATURES_LGB, train, val, test, args.finish_buffer_hours, base_unit)

    if "online" in units:
        for fold in range(6, 11):
            folds = np.array_split(np.arange(n), 10)
            test_fold = d.iloc[folds[fold - 1]].copy()
            test_start = test_fold.trigger_time.min()
            train_fold = d.iloc[np.concatenate(folds[:fold - 1])].loc[lambda x: x.finish_boundary < test_start]
            if train_fold.target_failure.nunique() < 2 or test_fold.target_failure.nunique() < 2:
                print(f"SKIP fold {fold}: one class"); continue
            base_unit = f"finance_fineract_online_fold_{fold}"
            if "rf" in models:
                run_model("rf", "C", lambda feats: rf(feats), FEATURES_RF, train_fold, None, test_fold, args.finish_buffer_hours, base_unit)
            if "lgbm" in models:
                run_model("lgbm", "E", None, FEATURES_LGB, train_fold, None, test_fold, args.finish_buffer_hours, base_unit)

    # Print summary
    print("\n── Summary ──")
    rows = []
    for js in sorted(PARTS.glob(f"*__finish_buffer_hours_{args.finish_buffer_hours}.json")):
        data_loaded = json.loads(js.read_text())
        if isinstance(data_loaded, list):
            rows.extend(data_loaded)
        else:
            rows.append(data_loaded)
    if rows:
        df = pd.DataFrame(rows)
        if "model" in df.columns and "roc_auc" in df.columns:
            for model_name, grp in df.groupby("model"):
                online = grp[grp.split.str.startswith("online")]
                final = grp[grp.split == "final_20pct"]
                macro_auc = online.roc_auc.mean() if len(online) else float("nan")
                macro_lift = online.lift_at_15.mean() if len(online) else float("nan")
                final_auc = final.roc_auc.iloc[0] if len(final) else float("nan")
                final_lift = final.lift_at_15.iloc[0] if len(final) else float("nan")
                print(f"  {model_name:6s}  online macro AUC={macro_auc:.3f}  lift@15%={macro_lift:.3f}  final AUC={final_auc:.3f}  lift@15%={final_lift:.3f}")


if __name__ == "__main__":
    main()
