"""Leakage-safe Apache Fineract finance-pilot evaluation (protocol v1.1).

This script uses pre-trigger commit-size/calendar fields plus local history
only from suites completed before the current trigger. It evaluates a fixed
chronological final-20% development test and five expanding online folds.
Calibration choice is made by Brier score on the earlier validation window;
the test period is never used to select calibration or a threshold.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROTOCOLS = {
    "v1_1": (ROOT / "data" / "processed" / "phase2_finance_v1_annotated_suites.parquet", REPORTS / "phase2_finance_v1_parts"),
    "v1_2": (ROOT / "data" / "processed" / "phase2_finance_v1_2_suites.parquet", REPORTS / "phase2_finance_v1_2_parts"),
}
DATA, PARTS = PROTOCOLS["v1_1"]
SEED = 20261013
STRICT = ["log_additions", "log_deletions", "log_total_changes", "log_changed_file_count", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
HISTORY = ["previous_completed_failure", "last5_completed_failure_rate", "all_prior_completed_failure_rate", "hours_since_completed", "hours_since_failure"]
FEATURES = STRICT + HISTORY
# v1.2 pre-specified secondary variant: history of the same branch (PR source or pushed branch).
STREAM = ["stream_previous_completed_failure", "stream_last5_completed_failure_rate", "stream_completed_count"]
VARIANTS = {"v1_1": [("C_rf", FEATURES)], "v1_2": [("C_rf", FEATURES), ("Cstream_rf", FEATURES + STREAM)]}


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
    edges = np.linspace(0, 1, bins + 1); total = len(y); value = 0.0
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
        q, r, l = top_metrics(y, p, pct / 100); result[f"precision_at_{pct}"]=q; result[f"recall_at_{pct}"]=r; result[f"lift_at_{pct}"]=l
    result["recall_at_15_cap"] = float(min(1.0, .15 / y.mean())) if y.mean() else np.nan
    return result


def add_safe_history(data: pd.DataFrame, buffer_hours: int) -> pd.DataFrame:
    d=data.sort_values(["trigger_time", "head_sha"], kind="mergesort").copy()
    d["finish_boundary"] = d.finish_time + pd.Timedelta(hours=buffer_hours)
    d["previous_completed_failure"] = 0.; d["last5_completed_failure_rate"] = 0.; d["all_prior_completed_failure_rate"] = 0.
    d["hours_since_completed"] = -1.; d["hours_since_failure"] = -1.
    has_stream = "stream" in d.columns
    if has_stream:
        d["stream_previous_completed_failure"] = -1.; d["stream_last5_completed_failure_rate"] = -1.; d["stream_completed_count"] = 0.
    completed=[]; pending=[]; cursor=0; by_stream: dict[str, list[int]] = {}
    # Each target is observed only after its conservative finish boundary.
    for idx, row in d.iterrows():
        pending.append((row.finish_boundary, row.trigger_time, row.head_sha, int(row.target_failure), idx))
    pending.sort(key=lambda x:(x[0],x[1],x[2]))
    for idx, row in d.iterrows():
        while cursor < len(pending) and pending[cursor][0] < row.trigger_time:
            completed.append(pending[cursor])
            if has_stream: by_stream.setdefault(str(d.at[pending[cursor][4], "stream"]), []).append(pending[cursor][3])
            cursor += 1
        if has_stream and (history := by_stream.get(str(row.stream))):
            d.at[idx,"stream_previous_completed_failure"] = history[-1]
            d.at[idx,"stream_last5_completed_failure_rate"] = float(np.mean(history[-5:]))
            d.at[idx,"stream_completed_count"] = float(len(history))
        if not completed: continue
        failures=[v[3] for v in completed]
        d.at[idx,"previous_completed_failure"] = failures[-1]
        d.at[idx,"last5_completed_failure_rate"] = float(np.mean(failures[-5:]))
        d.at[idx,"all_prior_completed_failure_rate"] = float(np.mean(failures))
        d.at[idx,"hours_since_completed"] = max(0., (row.trigger_time-completed[-1][0]).total_seconds()/3600)
        prior_fails=[v for v in completed if v[3] == 1]
        if prior_fails: d.at[idx,"hours_since_failure"] = max(0., (row.trigger_time-prior_fails[-1][0]).total_seconds()/3600)
    return d


def prep(data: pd.DataFrame) -> pd.DataFrame:
    d=data.copy(); d["trigger_time"]=pd.to_datetime(d.trigger_time,utc=True); d["finish_time"]=pd.to_datetime(d.finish_time,utc=True)
    for source, target in [("additions","log_additions"),("deletions","log_deletions"),("total_changes","log_total_changes"),("changed_file_count","log_changed_file_count")]: d[target]=np.log1p(d[source].astype(float))
    hour=d.trigger_time.dt.hour + d.trigger_time.dt.minute/60; dow=d.trigger_time.dt.dayofweek
    d["hour_sin"]=np.sin(2*np.pi*hour/24); d["hour_cos"]=np.cos(2*np.pi*hour/24); d["dow_sin"]=np.sin(2*np.pi*dow/7); d["dow_cos"]=np.cos(2*np.pi*dow/7)
    return d.sort_values(["trigger_time","head_sha"],kind="mergesort").reset_index(drop=True)


def rf() -> Pipeline:
    return Pipeline([("impute",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=200,max_depth=15,min_samples_split=5,min_samples_leaf=2,class_weight="balanced",random_state=SEED,n_jobs=-1))])


def logistic() -> Pipeline:
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(class_weight="balanced",random_state=SEED,max_iter=2000))])


def choose_calibration(raw_validation: np.ndarray, y_validation: np.ndarray) -> tuple[str, object | None, dict]:
    raw=np.clip(raw_validation,1e-6,1-1e-6); result={"raw_brier":float(brier_score_loss(y_validation,raw))}
    if len(np.unique(y_validation)) < 2: return "raw",None,result
    calibrator=LogisticRegression(C=1e6,random_state=SEED,max_iter=2000).fit(np.log(raw/(1-raw)).reshape(-1,1),y_validation)
    calibrated=calibrator.predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]; result["sigmoid_brier"]=float(brier_score_loss(y_validation,calibrated))
    return ("sigmoid",calibrator,result) if result["sigmoid_brier"] < result["raw_brier"] else ("raw",None,result)


def apply_calibration(p: np.ndarray, calibrator: object | None) -> np.ndarray:
    if calibrator is None: return p
    q=np.clip(p,1e-6,1-1e-6); return calibrator.predict_proba(np.log(q/(1-q)).reshape(-1,1))[:,1]


def weekly_ci(pred: pd.DataFrame, draws: int=1000) -> dict:
    d=pred.copy(); d["week"]=pd.to_datetime(d.trigger_time,utc=True).dt.to_period("W").astype(str); blocks=[x for _,x in d.groupby("week")]; rng=np.random.default_rng(SEED); rows=[]
    for _ in range(draws):
        sample=pd.concat([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))],ignore_index=True)
        if sample.label.nunique() < 2: continue
        m=metrics(sample.label.to_numpy(int),sample.probability.to_numpy(float),""," ")
        rows.append({k:m[k] for k in ("roc_auc","recall_at_15","precision_at_15","lift_at_15")})
    table=pd.DataFrame(rows); return {f"{col}_weekly_ci_low":float(table[col].quantile(.025)) for col in table} | {f"{col}_weekly_ci_high":float(table[col].quantile(.975)) for col in table}


def save(unit: str, row: dict, pred: pd.DataFrame, buffer: int) -> None:
    PARTS.mkdir(parents=True,exist_ok=True); js,pq=part(unit,buffer); pred.to_parquet(pq,index=False); js.write_text(json.dumps([row],indent=2,default=str)+"\n")


def main() -> None:
    global DATA, PARTS
    parser=argparse.ArgumentParser(); parser.add_argument("--finish-buffer-hours",type=int,choices=(0,1,6),default=0); parser.add_argument("--units",default="final,online"); parser.add_argument("--protocol",choices=tuple(PROTOCOLS),default="v1_1"); args=parser.parse_args()
    DATA, PARTS = PROTOCOLS[args.protocol]
    base=prep(pd.read_parquet(DATA)); d=add_safe_history(base,args.finish_buffer_hours); n=len(d); units=set(args.units.split(","))
    manifest={"study":f"phase2_finance_pilot_{args.protocol}","seed":SEED,"variants":{name:feats for name,feats in VARIANTS[args.protocol]},"excluded_descriptive_badges":True,"finish_buffer_hours":args.finish_buffer_hours,"rows":n,"git_commit":"unavailable","python":sys.version,"sklearn":sklearn.__version__,"platform":platform.platform()}
    suffix="" if args.protocol=="v1_1" else f"_{args.protocol}"
    (REPORTS/f"phase2_finance_evaluation_manifest{suffix}.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for variant, FEATURES in VARIANTS[args.protocol]:
        run_units(d, n, units, variant, FEATURES, args.finish_buffer_hours)


def run_units(d: pd.DataFrame, n: int, units: set, variant: str, FEATURES: list, buffer: int) -> None:
    class _Args: finish_buffer_hours = buffer
    args = _Args()
    if "final" in units:
        unit=f"finance_fineract_final_{variant}"; js,pq=part(unit,args.finish_buffer_hours)
        if js.exists() and pq.exists(): print(f"SKIP {unit}: checkpoint exists")
        else:
            cut_test=int(.8*n); cut_val=int(.8*cut_test); val_start=d.iloc[cut_val].trigger_time; test_start=d.iloc[cut_test].trigger_time
            train=d.iloc[:cut_val].loc[lambda x:x.finish_boundary < val_start]; val=d.iloc[cut_val:cut_test].loc[lambda x:x.finish_boundary < test_start]; test=d.iloc[cut_test:].copy()
            if train.target_failure.nunique()<2 or val.target_failure.nunique()<2 or test.target_failure.nunique()<2: raise ValueError("final split lacks both labels")
            model=rf().fit(train[FEATURES],train.target_failure); raw_val=model.predict_proba(val[FEATURES])[:,1]; choice,cal,cal_result=choose_calibration(raw_val,val.target_failure.to_numpy(int)); p=apply_calibration(model.predict_proba(test[FEATURES])[:,1],cal)
            pred=pd.DataFrame({"repo":test.repository,"build_id":test.head_sha,"trigger_time":test.trigger_time,"label":test.target_failure.astype(int),"probability":p,"fold_window":"final_20pct","model":"rf","tier":variant.split("_")[0],"calibration":choice})
            row=metrics(pred.label.to_numpy(int),pred.probability.to_numpy(float),"rf", "final_20pct"); row.update({"repository":"apache/fineract","finish_buffer_hours":args.finish_buffer_hours,"n_train":len(train),"n_validation":len(val),"n_test":len(test),"calibration":choice,**cal_result,**weekly_ci(pred)})
            save(unit,row,pred,args.finish_buffer_hours); print(f"DONE {unit} calibration={choice}")
    if "online" in units:
        for fold in range(6,11):
            unit=f"finance_fineract_online_fold_{fold}_{variant}"; js,pq=part(unit,args.finish_buffer_hours)
            if js.exists() and pq.exists(): print(f"SKIP {unit}: checkpoint exists"); continue
            folds=np.array_split(np.arange(n),10); test=d.iloc[folds[fold-1]].copy(); test_start=test.trigger_time.min(); train=d.iloc[np.concatenate(folds[:fold-1])].loc[lambda x:x.finish_boundary < test_start]
            if train.target_failure.nunique()<2 or test.target_failure.nunique()<2: print(f"SKIP {unit}: one class"); continue
            model=rf().fit(train[FEATURES],train.target_failure); p=model.predict_proba(test[FEATURES])[:,1]
            pred=pd.DataFrame({"repo":test.repository,"build_id":test.head_sha,"trigger_time":test.trigger_time,"label":test.target_failure.astype(int),"probability":p,"fold_window":f"online_fold_{fold}","model":"rf","tier":variant.split("_")[0],"calibration":"none"})
            row=metrics(pred.label.to_numpy(int),p,"rf",f"online_fold_{fold}"); row.update({"repository":"apache/fineract","finish_buffer_hours":args.finish_buffer_hours,"n_train":len(train),"n_test":len(test),**weekly_ci(pred)})
            save(unit,row,pred,args.finish_buffer_hours); print(f"DONE {unit}")


if __name__=="__main__": main()
