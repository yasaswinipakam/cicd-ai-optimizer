"""Prediction, build-history, and dashboard API endpoints."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session, joinedload

from api.db.database import get_db
from api.demo_mode import read_only
from api.db.models import BuildLog, JobFailureAnalysis, JobRun, Pipeline, Prediction, WorkflowRun
from api.schemas import (
    BuildCreatedOut,
    BuildFeaturesIn,
    AlertThresholdOut,
    BuildLogOut,
    DashboardSummaryOut,
    FailureTrendOut,
    JobFailureAnalysisOut,
    JobFailureLogIn,
    JobTelemetryOut,
    ModelInfoOut,
    OperationalDashboardOut,
    PipelineOut,
    RiskDistributionOut,
    ShadowBuildCreatedOut,
    ShadowOutcomeIn,
)
from src.features.stream_context import PriorBuild, gha_stream_key, stream_context, stream_key
from src.inference import alert_policy
from src.inference.predictor import CICDPredictor
from api.diagnostics_service import save_job_failure_analysis
from src.models.suggestions import get_suggestion
from src.reports.pdf_generator import generate_build_report_pdf


router = APIRouter(tags=["pipelines"])


@lru_cache
def get_predictor() -> CICDPredictor:
    """Lazily load heavyweight ML artifacts on the first prediction request."""
    try:
        return CICDPredictor()
    except (FileNotFoundError, OSError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction artifacts are unavailable; complete model training first.",
        ) from error


def _find_or_create_pipeline(identifier: str, db: Session) -> Pipeline:
    """Resolve a numeric primary key or human-readable pipeline name."""
    pipeline = db.query(Pipeline).filter(Pipeline.name == identifier).first()
    if pipeline is None and identifier.isdigit():
        pipeline = db.get(Pipeline, int(identifier))
    if pipeline is None:
        pipeline = Pipeline(name=identifier, status="active")
        db.add(pipeline)
        db.flush()
    return pipeline


# A shadow build still "running" after this long almost surely never had its
# outcome reported; counting it as in flight forever would skew the context.
STALE_PENDING_HOURS = 24


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _observed_outcome(build: BuildLog) -> tuple[datetime | None, bool | None]:
    """When this build's real CI outcome became known, and whether it failed.

    Builds submitted without an observed status only store the model's own
    prediction, which must never be fed back in as history.
    """
    raw = build.raw_features or {}
    if build.build_status not in ("passed", "failed"):
        return None, None
    if raw.get("_outcome_recorded_at"):
        return _as_utc(datetime.fromisoformat(raw["_outcome_recorded_at"])), build.build_status == "failed"
    if raw.get("_observed_outcome"):
        return _as_utc(build.timestamp), build.build_status == "failed"
    return None, None


def _with_stream_context(
    predictor: CICDPredictor, db: Session, pipeline: Pipeline, features: dict, now: datetime
) -> tuple[dict, dict | None]:
    """Add Tier E history and in-flight context derived from this pipeline's stored builds.

    The server derives these instead of trusting client-sent values, using
    the training rules in ``src.features.stream_context``: the stream is the
    PR (``gh_pull_req_num``) or the branch, outcomes count once reported, and
    shadow builds awaiting an outcome count as running.
    """
    if not getattr(predictor, "uses_stream_context", False):
        return features, None
    github = getattr(predictor, "input_schema", "travis") == "github_actions"

    def key(raw: dict) -> str:
        if github:
            return gha_stream_key(raw.get("event"), raw.get("actor"), raw.get("head_branch"))
        return stream_key(raw.get("gh_is_pr"), raw.get("gh_pull_req_num"), raw.get("git_branch"))

    def previous_of(raw: dict) -> tuple:
        # Travis: the latest earlier build on the branch.  GitHub Actions: the
        # latest earlier run of the same workflow on the same stream.
        if github:
            return str(raw.get("workflow_id") or raw.get("workflow") or ""), key(raw)
        return (str(raw.get("git_branch") or ""),)

    current, current_previous = key(features), previous_of(features)
    prior, previous_build_id = [], None
    builds = (
        db.query(BuildLog).filter(BuildLog.pipeline_id == pipeline.id).order_by(BuildLog.timestamp, BuildLog.id).all()
    )
    for build in builds:
        raw = build.raw_features or {}
        started = _as_utc(build.timestamp)
        finished, failed = _observed_outcome(build)
        if finished is None:
            awaiting = (raw.get("_prediction_context") or {}).get("shadow_mode") and build.build_status == "pending"
            if not awaiting or now - started > timedelta(hours=STALE_PENDING_HOURS):
                continue
        prior.append(PriorBuild(build.id, started, key(raw), finished, failed))
        if previous_of(raw) == current_previous:
            previous_build_id = build.id
    context = stream_context(prior, started_at=now, previous_build_id=previous_build_id, stream=current)
    if github:
        context["time_since_last_commit"] = _seconds_since_previous_commit(builds, features)
    json_safe = {key: (None if value != value else float(value)) for key, value in context.items()}
    return {**features, **context}, json_safe


def _seconds_since_previous_commit(builds: list[BuildLog], features: dict) -> float:
    """Seconds from the pipeline's latest earlier commit to this one, by commit date.

    Training computes this over the repository's collected commits, i.e. the
    commits that triggered runs; here those are the commits of stored builds.
    Unknown (NaN) without a commit date or an earlier commit.
    """
    committed = pd.to_datetime(features.get("committed_at"), errors="coerce", utc=True)
    if pd.isna(committed):
        value = pd.to_numeric(pd.Series([features.get("time_since_last_commit")]), errors="coerce").iloc[0]
        return float(value)
    earlier = [
        stamp for build in builds
        if (raw := build.raw_features or {}).get("head_sha") != features.get("head_sha")
        and not pd.isna(stamp := pd.to_datetime(raw.get("committed_at"), errors="coerce", utc=True))
        and stamp <= committed
    ]
    return (committed - max(earlier)).total_seconds() if earlier else float("nan")


def _alert_threshold(predictor: CICDPredictor, db: Session, pipeline: Pipeline) -> tuple[float, str, int]:
    """This pipeline's alert threshold for its next build (see ``src.inference.alert_policy``).

    Returns the threshold, its basis and how many earlier scores it used.
    """
    recent = (
        db.query(Prediction.failure_probability)
        .join(BuildLog, Prediction.build_log_id == BuildLog.id)
        .filter(BuildLog.pipeline_id == pipeline.id)
        .order_by(BuildLog.timestamp.desc(), BuildLog.id.desc())
        .limit(alert_policy.WINDOW)
        .all()
    )
    scores = [probability for (probability,) in reversed(recent)]
    # The model's threshold is the typical-repository fallback; legacy stubs default to 0.5.
    threshold, basis = alert_policy.repository_threshold(scores, getattr(predictor, "decision_threshold", 0.5))
    return threshold, basis, len(scores)


def _apply_alert_policy(prediction_data: dict, threshold: float) -> None:
    """Replace the model's global-threshold verdict with the pipeline's review-budget one."""
    prediction_data["predicted_status"] = "failed" if prediction_data["failure_probability"] >= threshold else "passed"


def _prediction_out(prediction: Prediction) -> dict:
    return {
        "failure_probability": prediction.failure_probability,
        "predicted_status": prediction.predicted_status,
        "confidence": max(prediction.failure_probability, 1 - prediction.failure_probability),
        "is_anomaly": prediction.is_anomaly,
        "anomaly_score": prediction.anomaly_score,
        "root_cause_cluster_id": prediction.root_cause_cluster_id,
        "root_cause_label": prediction.root_cause_label,
    }


@router.post("/pipelines/{pipeline_id:path}/builds", response_model=BuildCreatedOut)
def create_build(
    pipeline_id: str,
    payload: BuildFeaturesIn,
    explain: bool = Query(False),
    db: Session = Depends(get_db),
    predictor: CICDPredictor = Depends(get_predictor),
) -> dict:
    """Analyze, persist, and return one new build for a pipeline."""
    pipeline = _find_or_create_pipeline(pipeline_id, db)
    now = datetime.now(timezone.utc)
    features, context = _with_stream_context(predictor, db, pipeline, payload.build_features, now)
    threshold, basis, _ = _alert_threshold(predictor, db, pipeline)
    analysis = (
        predictor.full_analysis(features, payload.failed_test_log, include_explanation=True)
        if explain
        else predictor.full_analysis(features, payload.failed_test_log)
    )
    prediction_data = analysis["prediction"]
    _apply_alert_policy(prediction_data, threshold)
    anomaly_data = analysis["anomaly"]
    root_cause = analysis.get("root_cause") or {}
    # Without a reported status the outcome is unknown; never store the prediction as if observed.
    observed_status = payload.build_status or "unknown"
    build_log = BuildLog(
        pipeline_id=pipeline.id,
        timestamp=now,
        log_content=payload.failed_test_log,
        build_status=observed_status,
        stage_durations=payload.stage_durations,
        raw_features={
            **payload.build_features,
            "_explanation": analysis.get("explanation", []),
            # Only a reported status is a real outcome usable as later history.
            "_observed_outcome": payload.build_status is not None,
            "_stream_context": context,
        },
    )
    db.add(build_log)
    db.flush()
    prediction = Prediction(
        build_log_id=build_log.id,
        model_name=predictor.best_model_name,
        failure_probability=prediction_data["failure_probability"],
        predicted_status=prediction_data["predicted_status"],
        is_anomaly=anomaly_data["is_anomaly"],
        anomaly_score=anomaly_data["anomaly_score"],
        root_cause_cluster_id=root_cause.get("cluster_id"),
        root_cause_label=root_cause.get("likely_cause"),
    )
    db.add(prediction)
    db.flush()
    result = _prediction_out(prediction)
    result["explanation"] = analysis.get("explanation") if explain else None
    result.update({"build_log_id": build_log.id, "pipeline_id": pipeline.id, "pipeline_name": pipeline.name,
                   "stream_context": context, "decision_threshold": threshold, "alert_basis": basis,
                   "is_alert": prediction_data["failure_probability"] >= threshold})
    if read_only():
        db.rollback()  # the public demo scores against stored history but keeps nothing
    else:
        db.commit()
    return result


@router.post("/pipelines/{pipeline_id:path}/shadow-builds", response_model=ShadowBuildCreatedOut)
def create_shadow_build(
    pipeline_id: str,
    payload: BuildFeaturesIn,
    explain: bool = Query(False),
    db: Session = Depends(get_db),
    predictor: CICDPredictor = Depends(get_predictor),
) -> dict:
    """Persist a non-blocking pre-build prediction that awaits an actual CI outcome.

    This endpoint is intentionally shadow-only: it does not reject, cancel, or
    alter any CI build. Use the outcome endpoint after the CI provider reports
    `passed` or `failed` to create an auditable live-evaluation record.
    """
    pipeline = _find_or_create_pipeline(pipeline_id, db)
    now = datetime.now(timezone.utc)
    features, context = _with_stream_context(predictor, db, pipeline, payload.build_features, now)
    threshold, basis, _ = _alert_threshold(predictor, db, pipeline)
    analysis = predictor.full_analysis(features, include_explanation=explain)
    prediction_data = analysis["prediction"]
    _apply_alert_policy(prediction_data, threshold)
    anomaly_data = analysis["anomaly"]
    model_context = {
        "shadow_mode": True,
        "model_name": predictor.best_model_name,
        "model_artifact_sha256": predictor.model_artifact_sha256,
        "calibration_method": predictor.calibration_method,
        "calibration_artifact_sha256": predictor.calibrator_artifact_sha256,
        "training_mode": predictor.preprocessing.get("training_mode"),
        "feature_schema": predictor.feature_cols,
        "decision_threshold": threshold,
        "alert_basis": basis,
        "prediction_created_at": now.isoformat(),
    }
    build_log = BuildLog(
        pipeline_id=pipeline.id,
        timestamp=now,
        log_content=None,
        build_status="pending",
        stage_durations={},
        raw_features={
            **payload.build_features,
            "_explanation": analysis.get("explanation", []),
            "_prediction_context": model_context,
            "_stream_context": context,
        },
    )
    db.add(build_log)
    db.flush()
    prediction = Prediction(
        build_log_id=build_log.id,
        model_name=predictor.best_model_name,
        failure_probability=prediction_data["failure_probability"],
        predicted_status=prediction_data["predicted_status"],
        is_anomaly=anomaly_data["is_anomaly"],
        anomaly_score=anomaly_data["anomaly_score"],
        root_cause_cluster_id=None,
        root_cause_label=None,
    )
    db.add(prediction)
    db.commit()
    result = _prediction_out(prediction)
    result["explanation"] = analysis.get("explanation") if explain else None
    result.update(
        {
            "build_log_id": build_log.id,
            "pipeline_id": pipeline.id,
            "pipeline_name": pipeline.name,
            "decision_threshold": threshold,
            "alert_basis": basis,
            "is_alert": prediction_data["failure_probability"] >= threshold,
            "model_version": predictor.model_artifact_sha256,
            "training_mode": predictor.preprocessing.get("training_mode"),
            "calibration_method": predictor.calibration_method,
            "calibration_version": predictor.calibrator_artifact_sha256,
            "stream_context": context,
        }
    )
    return result


@router.patch("/pipelines/{pipeline_id:path}/shadow-builds/{build_id}/outcome", response_model=BuildLogOut)
def record_shadow_outcome(
    pipeline_id: str,
    build_id: int,
    payload: ShadowOutcomeIn,
    db: Session = Depends(get_db),
    predictor: CICDPredictor = Depends(get_predictor),
) -> dict:
    """Attach the observed CI result to a shadow record without rescoring it."""
    pipeline = db.query(Pipeline).filter(Pipeline.name == pipeline_id).first()
    if pipeline is None and pipeline_id.isdigit():
        pipeline = db.get(Pipeline, int(pipeline_id))
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    build_log = (
        db.query(BuildLog)
        .options(joinedload(BuildLog.predictions))
        .filter(BuildLog.id == build_id, BuildLog.pipeline_id == pipeline.id)
        .first()
    )
    if build_log is None:
        raise HTTPException(status_code=404, detail="Shadow build not found")
    context = (build_log.raw_features or {}).get("_prediction_context", {})
    if not context.get("shadow_mode"):
        raise HTTPException(status_code=409, detail="Build was not created in shadow mode")
    if build_log.build_status != "pending":
        raise HTTPException(status_code=409, detail="Observed outcome has already been recorded")
    build_log.build_status = payload.build_status
    build_log.log_content = payload.failed_test_log
    build_log.stage_durations = payload.stage_durations
    build_log.raw_features = {
        **(build_log.raw_features or {}),
        "_outcome_recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    prediction = build_log.predictions[0] if build_log.predictions else None
    if prediction and payload.build_status == "failed" and payload.failed_test_log:
        root_cause = predictor.analyze_root_cause(payload.failed_test_log)
        if root_cause:
            prediction.root_cause_cluster_id = root_cause["cluster_id"]
            prediction.root_cause_label = root_cause["likely_cause"]
    db.commit()
    return {
        "id": build_log.id,
        "timestamp": build_log.timestamp,
        "build_status": build_log.build_status,
        "failure_probability": prediction.failure_probability if prediction else None,
        "predicted_status": prediction.predicted_status if prediction else None,
        "is_anomaly": prediction.is_anomaly if prediction else None,
        "anomaly_score": prediction.anomaly_score if prediction else None,
        "root_cause_label": prediction.root_cause_label if prediction else None,
        "optimization_suggestion": None,
        "explanation": build_log.raw_features.get("_explanation") if build_log.raw_features else None,
    }


SPARKLINE_BUILDS = 12


@router.get("/model", response_model=ModelInfoOut)
def model_info(predictor: CICDPredictor = Depends(get_predictor)) -> dict:
    """Describe the deployed failure model, its alert threshold and its held-out results."""
    metadata = getattr(getattr(predictor, "model", None), "metadata", None) or {}
    return {
        "model_name": predictor.best_model_name,
        "training_mode": predictor.preprocessing.get("training_mode"),
        "decision_threshold": predictor.decision_threshold,
        "alert_policy": {"type": "repository_budget", "review_budget": alert_policy.REVIEW_BUDGET,
                         "window": alert_policy.WINDOW, "min_history": alert_policy.MIN_HISTORY},
        "input_schema": getattr(predictor, "input_schema", None),
        "holdout_metrics": metadata.get("holdout_metrics"),
        "new_repository_metrics": metadata.get("new_repository_holdout_metrics"),
    }


@router.get("/pipelines/{pipeline_id:path}/alert-threshold", response_model=AlertThresholdOut)
def pipeline_alert_threshold(
    pipeline_id: str, db: Session = Depends(get_db), predictor: CICDPredictor = Depends(get_predictor)
) -> dict:
    """The alert threshold this pipeline's next build would get under its review budget."""
    pipeline = db.query(Pipeline).filter(Pipeline.name == pipeline_id).first()
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    threshold, basis, scored = _alert_threshold(predictor, db, pipeline)
    return {"threshold": threshold, "basis": basis, "scored_builds": scored,
            "review_budget": alert_policy.REVIEW_BUDGET, "min_history": alert_policy.MIN_HISTORY}


@router.get("/pipelines", response_model=list[PipelineOut])
def list_pipelines(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict]:
    """List pipelines and their most recently analyzed build."""
    pipelines = db.query(Pipeline).order_by(Pipeline.created_at.desc()).offset(offset).limit(limit).all()
    output = []
    for pipeline in pipelines:
        recent = (
            db.query(BuildLog)
            .options(joinedload(BuildLog.predictions))
            .filter(BuildLog.pipeline_id == pipeline.id)
            .order_by(BuildLog.timestamp.desc(), BuildLog.id.desc())
            .limit(SPARKLINE_BUILDS)
            .all()
        )
        latest = recent[0] if recent else None
        latest_prediction = latest.predictions[0] if latest and latest.predictions else None
        output.append(
            {
                "id": pipeline.id,
                "name": pipeline.name,
                "status": pipeline.status,
                "latest_build_status": latest.build_status if latest else None,
                "latest_failure_probability": latest_prediction.failure_probability if latest_prediction else None,
                # Oldest first, for the per-pipeline risk sparkline.
                "recent_failure_probabilities": [
                    log.predictions[0].failure_probability for log in reversed(recent) if log.predictions
                ],
            }
        )
    return output


@router.get("/pipelines/{pipeline_id:path}/builds", response_model=list[BuildLogOut])
def list_builds(
    pipeline_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict]:
    """List one pipeline's build history, newest first."""
    pipeline = db.query(Pipeline).filter(Pipeline.name == pipeline_id).first()
    if pipeline is None and pipeline_id.isdigit():
        pipeline = db.get(Pipeline, int(pipeline_id))
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    logs = (
        db.query(BuildLog)
        .options(joinedload(BuildLog.predictions))
        .filter(BuildLog.pipeline_id == pipeline.id)
        .order_by(BuildLog.timestamp.desc(), BuildLog.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": log.id,
            "timestamp": log.timestamp,
            "build_status": log.build_status,
            "failure_probability": log.predictions[0].failure_probability if log.predictions else None,
            "predicted_status": log.predictions[0].predicted_status if log.predictions else None,
            "is_anomaly": log.predictions[0].is_anomaly if log.predictions else None,
            "anomaly_score": log.predictions[0].anomaly_score if log.predictions else None,
            "root_cause_label": log.predictions[0].root_cause_label if log.predictions else None,
            "optimization_suggestion": (
                get_suggestion(
                    (log.predictions[0].root_cause_label or "").split(","),
                    is_anomaly=log.predictions[0].is_anomaly,
                    anomaly_score=log.predictions[0].anomaly_score,
                )
                # Without a failed-test log there is no root cause, and the suggestion would be generic filler.
                if log.predictions and log.build_status == "failed" and log.predictions[0].root_cause_label
                else None
            ),
            "explanation": log.raw_features.get("_explanation") if log.raw_features else None,
        }
        for log in logs
    ]


@router.get("/pipelines/{pipeline_id:path}/telemetry/jobs", response_model=list[JobTelemetryOut])
def list_job_telemetry(
    pipeline_id: str,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Return persisted runtime telemetry for bottleneck/stall visualization.

    These observations are collected during/after CI execution and deliberately
    stay out of the pre-build risk model.
    """
    pipeline = db.query(Pipeline).filter(Pipeline.name == pipeline_id).first()
    if pipeline is None and pipeline_id.isdigit():
        pipeline = db.get(Pipeline, int(pipeline_id))
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    jobs = (
        db.query(JobRun)
        .join(WorkflowRun)
        .filter(WorkflowRun.pipeline_id == pipeline.id)
        .order_by(JobRun.updated_at.desc(), JobRun.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "job_run_id": job.id,
            "workflow_run_id": job.workflow_run_id,
            "workflow_name": job.workflow_run.workflow_name,
            "job_name": job.name,
            "status": job.status,
            "conclusion": job.conclusion,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "duration_seconds": job.duration_seconds,
            "bottleneck_status": job.bottleneck_status,
            "baseline_sample_size": job.baseline_sample_size,
            "baseline_median_seconds": job.baseline_median_seconds,
            "baseline_p95_seconds": job.baseline_p95_seconds,
        }
        for job in jobs
    ]


@router.post(
    "/pipelines/{pipeline_id:path}/telemetry/jobs/{job_run_id}/failure-log",
    response_model=JobFailureAnalysisOut,
)
def ingest_job_failure_log(
    pipeline_id: str,
    job_run_id: int,
    payload: JobFailureLogIn,
    db: Session = Depends(get_db),
) -> dict:
    """Store and classify an actual failed-job log after redaction.

    A GitHub Actions webhook does not include console output.  A trusted
    collector must retrieve the log with its own Actions-read permission and
    send it here.  This endpoint is post-failure only and never invokes the
    predictor or modifies a BuildLog/Prediction feature record.
    """
    pipeline = db.query(Pipeline).filter(Pipeline.name == pipeline_id).first()
    if pipeline is None and pipeline_id.isdigit():
        pipeline = db.get(Pipeline, int(pipeline_id))
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    job = (
        db.query(JobRun)
        .join(WorkflowRun)
        .filter(JobRun.id == job_run_id, WorkflowRun.pipeline_id == pipeline.id)
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job telemetry not found")
    if str(job.conclusion or "").lower() not in {"failure", "timed_out", "cancelled", "action_required"}:
        raise HTTPException(status_code=409, detail="Failure logs may be attached only to a failed job")

    analysis = save_job_failure_analysis(job, payload.log_content, payload.log_source, db)
    return {
        "job_run_id": job.id,
        "category": analysis.category,
        "title": analysis.title,
        "summary": analysis.summary,
        "signature": analysis.signature,
        "matched_signals": analysis.matched_signals,
        "created_at": analysis.created_at,
    }


@router.get(
    "/pipelines/{pipeline_id:path}/telemetry/jobs/{job_run_id}/failure-analysis",
    response_model=JobFailureAnalysisOut,
)
def get_job_failure_analysis(
    pipeline_id: str,
    job_run_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Return a stored post-failure diagnostic without exposing raw log text."""
    analysis = (
        db.query(JobFailureAnalysis)
        .join(JobRun)
        .join(WorkflowRun)
        .filter(JobFailureAnalysis.job_run_id == job_run_id, WorkflowRun.pipeline.has(name=pipeline_id))
        .first()
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="Failure analysis not found")
    return {
        "job_run_id": analysis.job_run_id,
        "category": analysis.category,
        "title": analysis.title,
        "summary": analysis.summary,
        "signature": analysis.signature,
        "matched_signals": analysis.matched_signals,
        "created_at": analysis.created_at,
    }


@router.get("/pipelines/{pipeline_id:path}/builds/{build_id}/report.pdf")
def download_build_report(
    pipeline_id: str,
    build_id: int,
    db: Session = Depends(get_db),
) -> Response:
    """Generate and return a temporary PDF report for one persisted build."""
    pipeline = db.query(Pipeline).filter(Pipeline.name == pipeline_id).first()
    if pipeline is None and pipeline_id.isdigit():
        pipeline = db.get(Pipeline, int(pipeline_id))
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    build_log = (
        db.query(BuildLog)
        .options(joinedload(BuildLog.predictions))
        .filter(BuildLog.id == build_id, BuildLog.pipeline_id == pipeline.id)
        .first()
    )
    if build_log is None:
        raise HTTPException(status_code=404, detail="Build not found")
    prediction = build_log.predictions[0] if build_log.predictions else None
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found for build")
    top_terms = (prediction.root_cause_label or "").split(",") if prediction.root_cause_label else []
    suggestion = (
        get_suggestion(top_terms, is_anomaly=prediction.is_anomaly, anomaly_score=prediction.anomaly_score)
        if build_log.build_status == "failed" and top_terms else None
    )
    build_data = {
        "pipeline_id": pipeline.name,
        "build_id": build_log.id,
        "timestamp": build_log.timestamp.isoformat(),
        "build_status": build_log.build_status,
        "prediction": _prediction_out(prediction),
        "anomaly": {"is_anomaly": prediction.is_anomaly, "anomaly_score": prediction.anomaly_score},
        "root_cause": {"cluster_id": prediction.root_cause_cluster_id, "likely_cause": prediction.root_cause_label, "top_terms": top_terms},
        "top_terms": top_terms,
        "optimization_suggestion": suggestion,
        # Explanations are stored with the submitted build so reports reproduce
        # the local attribution available when that prediction was made.
        "explanation": build_log.raw_features.get("_explanation", []) if build_log.raw_features else [],
    }
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        generate_build_report_pdf(build_data, str(temporary_path))
        content = temporary_path.read_bytes()
    finally:
        temporary_path.unlink(missing_ok=True)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="build_{build_id}_report.pdf"'},
    )


@router.get("/dashboard/summary", response_model=DashboardSummaryOut)
def dashboard_summary(
    db: Session = Depends(get_db)
) -> dict:
    """Return system-wide prediction and root-cause aggregates."""
    logs = db.query(BuildLog).options(joinedload(BuildLog.predictions)).all()
    predictions = [prediction for log in logs for prediction in log.predictions]
    total = len(logs)
    outcomes = [failed for log in logs if (failed := _observed_outcome(log)[1]) is not None]
    failure_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
    anomaly_count = sum(prediction.is_anomaly for prediction in predictions)
    grouped: dict[tuple[int, str | None], int] = {}
    for prediction in predictions:
        if prediction.root_cause_cluster_id is not None:
            key = (prediction.root_cause_cluster_id, prediction.root_cause_label)
            grouped[key] = grouped.get(key, 0) + 1
    top_causes = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:3]
    return {
        "total_builds_analyzed": total,
        "pipelines_monitored": db.query(Pipeline).count(),
        "overall_failure_rate": failure_rate,
        "builds_with_outcome": len(outcomes),
        "anomalies_detected": anomaly_count,
        "top_root_causes": [
            {"cluster_id": cluster_id, "label": label, "count": count}
            for ((cluster_id, label), count) in top_causes
        ],
    }


@router.get("/dashboard/risk-distribution", response_model=RiskDistributionOut)
def risk_distribution(
    threshold: float | None = Query(None, ge=0, le=1), db: Session = Depends(get_db)
) -> dict:
    """Count predictions in ten equal failure-probability bins, and how many were alerts.

    Alerts are those raised when each build was scored, under its pipeline's
    review budget; with ``threshold``, those at or above it instead.
    """
    counts = [0] * 10
    rows = db.query(Prediction.failure_probability, Prediction.predicted_status).all()
    for probability, _ in rows:
        counts[min(int(probability * 10), 9)] += 1
    flagged = (sum(status == "failed" for _, status in rows) if threshold is None
               else sum(probability >= threshold for probability, _ in rows))
    return {
        "bins": [{"start": i / 10, "end": (i + 1) / 10, "count": count} for i, count in enumerate(counts)],
        "total": len(rows),
        "flagged": flagged,
    }


@router.get("/dashboard/failure-trends", response_model=list[FailureTrendOut])
def failure_trends(
    days: int = Query(30, ge=1, le=365),
    granularity: Literal["day", "build"] = Query("day"),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Failure risk and outcomes over time.

    ``day``: per day, the average predicted risk of scored builds and the
    failure rate over builds with a reported outcome (null when none).
    ``build``: one point per scored build in time order, with its predicted
    risk and the running failure rate of reported outcomes so far; useful
    while there are too few days for a daily trend.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    logs = (
        db.query(BuildLog).options(joinedload(BuildLog.predictions))
        .filter(BuildLog.timestamp >= since).order_by(BuildLog.timestamp, BuildLog.id).all()
    )
    if granularity == "build":
        points, reported, failed_so_far = [], 0, 0
        for log in logs:
            failed = _observed_outcome(log)[1]
            if failed is not None:
                reported += 1; failed_so_far += failed
            if log.predictions:
                points.append({
                    "date": _as_utc(log.timestamp).isoformat(),
                    "total_builds": reported, "failed_builds": failed_so_far,
                    "failure_rate": failed_so_far / reported if reported else None,
                    "scored_builds": 1, "average_predicted_risk": log.predictions[0].failure_probability,
                })
        return points
    risks: dict[str, list[float]] = {}
    outcomes: dict[str, list[bool]] = {}
    for log in logs:
        day = log.timestamp.date().isoformat()
        if log.predictions:
            risks.setdefault(day, []).append(log.predictions[0].failure_probability)
        failed = _observed_outcome(log)[1]
        if failed is not None:
            outcomes.setdefault(day, []).append(failed)
    days_seen = sorted(set(risks) | set(outcomes))
    return [
        {
            "date": day,
            "total_builds": len(outcomes.get(day, [])),
            "failed_builds": sum(outcomes.get(day, [])),
            "failure_rate": sum(outcomes[day]) / len(outcomes[day]) if day in outcomes else None,
            "scored_builds": len(risks.get(day, [])),
            "average_predicted_risk": sum(risks[day]) / len(risks[day]) if day in risks else None,
        }
        for day in days_seen
    ]


@router.get("/dashboard/operational-summary", response_model=OperationalDashboardOut)
def operational_dashboard_summary(
    db: Session = Depends(get_db)
) -> dict:
    """Aggregate runtime bottleneck and post-failure evidence across pipelines.

    These are operational/post-execution measurements. They intentionally do
    not contribute to the pre-build prediction model or its dashboard metrics.
    """
    jobs = db.query(JobRun).all()
    grouped_jobs: dict[str, dict] = {}
    for job in jobs:
        item = grouped_jobs.setdefault(
            job.name,
            {"job_name": job.name, "count": 0, "bottleneck_or_stall_count": 0, "durations": []},
        )
        item["count"] += 1
        if job.bottleneck_status in {"bottleneck", "stalled"}:
            item["bottleneck_or_stall_count"] += 1
        if job.duration_seconds is not None:
            item["durations"].append(float(job.duration_seconds))
    job_types = [
        {
            "job_name": item["job_name"],
            "count": item["count"],
            "bottleneck_or_stall_count": item["bottleneck_or_stall_count"],
            "average_duration_seconds": (sum(item["durations"]) / len(item["durations"]) if item["durations"] else None),
        }
        for item in grouped_jobs.values()
    ]
    job_types.sort(
        key=lambda item: (item["bottleneck_or_stall_count"], item["average_duration_seconds"] or 0, item["count"]),
        reverse=True,
    )
    analyses = db.query(JobFailureAnalysis).all()
    grouped_failures: dict[tuple[str, str], int] = {}
    for analysis in analyses:
        key = (analysis.category, analysis.title)
        grouped_failures[key] = grouped_failures.get(key, 0) + 1
    categories = [
        {"category": category, "title": title, "count": count}
        for (category, title), count in sorted(grouped_failures.items(), key=lambda item: item[1], reverse=True)[:5]
    ]
    return {
        "total_jobs_observed": len(jobs),
        "bottlenecks_detected": sum(job.bottleneck_status == "bottleneck" for job in jobs),
        "stalled_jobs_detected": sum(job.bottleneck_status == "stalled" for job in jobs),
        "failed_jobs_with_captured_logs": len(analyses),
        "top_slowest_or_bottlenecked_jobs": job_types[:5],
        "top_failure_categories": categories,
    }
