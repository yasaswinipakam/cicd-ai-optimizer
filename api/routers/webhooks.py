"""Secret-protected CI provider webhook ingestion endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.db.database import get_db
from api.db.models import JobRun, JobStepRun, Pipeline, ProviderWorkflowOutcome, WorkflowRun
from api.diagnostics_service import save_job_failure_analysis
from api.routers.pipelines import _find_or_create_pipeline, create_build, get_predictor
from api.schemas import BuildFeaturesIn, GitHubJobLogCollectorIn, GitHubJobTelemetryCollectorIn, JobFailureAnalysisOut
from src.inference.predictor import CICDPredictor
from src.models.bottleneck import assess_job_timing


router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def verify_github_signature(payload_bytes: bytes, signature_header: str | None, secret: str) -> bool:
    """Verify GitHub's HMAC-SHA256 webhook signature in constant time."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.removeprefix("sha256="))


def verify_webhook_secret(x_webhook_secret: str | None = Header(default=None)) -> None:
    """Require the configured constant-time-compared webhook shared secret."""
    expected = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not expected or not x_webhook_secret or not hmac.compare_digest(x_webhook_secret, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook secret")


def verify_collector_secret(x_ci_diagnostics_secret: str | None = Header(default=None)) -> None:
    """Authenticate the optional Actions log collector with a separate secret."""
    expected = os.getenv("CI_DIAGNOSTICS_COLLECTOR_SECRET")
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Log collector is not configured")
    if not x_ci_diagnostics_secret or not hmac.compare_digest(x_ci_diagnostics_secret, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid diagnostics collector secret")


def _parse_provider_time(value: Any) -> datetime | None:
    """Parse provider ISO timestamps as timezone-aware UTC datetimes."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize SQLite-returned naive datetimes and provider times to UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _pipeline_name_from_github(payload: dict[str, Any], fallback: str) -> str:
    repository = payload.get("repository") or {}
    if isinstance(repository, str):
        return repository
    return repository.get("full_name") or repository.get("name") or fallback


def _duration_seconds(started_at: datetime | None, completed_at: datetime | None) -> float | None:
    started_at, completed_at = _as_utc(started_at), _as_utc(completed_at)
    if not started_at or not completed_at:
        return None
    seconds = (completed_at - started_at).total_seconds()
    return float(seconds) if seconds >= 0 else None


_STATUS_ORDER = {"queued": 0, "waiting": 0, "in_progress": 1, "completed": 2}


def _merge_status(existing: str | None, incoming: Any) -> str:
    """Keep lifecycle state monotonic when GitHub deliveries arrive late."""
    candidate = str(incoming or "").lower()
    current = str(existing or "").lower()
    if not candidate:
        return current or "queued"
    return candidate if _STATUS_ORDER.get(candidate, -1) >= _STATUS_ORDER.get(current, -1) else current


def _earliest(current: datetime | None, incoming: datetime | None) -> datetime | None:
    current, incoming = _as_utc(current), _as_utc(incoming)
    if current is None:
        return incoming
    if incoming is None:
        return current
    return min(current, incoming)


def _latest(current: datetime | None, incoming: datetime | None) -> datetime | None:
    current, incoming = _as_utc(current), _as_utc(incoming)
    if current is None:
        return incoming
    if incoming is None:
        return current
    return max(current, incoming)


def _upsert_job_steps(job_run: JobRun, steps: list[dict[str, Any]], db: Session) -> None:
    """Persist provider steps idempotently for post-run diagnostics only."""
    existing = {step.provider_step_number: step for step in job_run.steps}
    for index, payload in enumerate(steps, start=1):
        number = int(payload.get("number") or index)
        started_at = _parse_provider_time(payload.get("started_at"))
        completed_at = _parse_provider_time(payload.get("completed_at"))
        step = existing.get(number)
        if step is None:
            step = JobStepRun(job_run=job_run, provider_step_number=number, name=str(payload.get("name") or f"step-{number}"))
            db.add(step)
        step.name = str(payload.get("name") or step.name)
        step.status = _merge_status(step.status, payload.get("status"))
        if payload.get("conclusion"):
            step.conclusion = payload.get("conclusion")
        step.started_at = _earliest(step.started_at, started_at)
        step.completed_at = _latest(step.completed_at, completed_at)
        step.duration_seconds = _duration_seconds(step.started_at, step.completed_at)
        step.raw_payload = payload


def _github_job_telemetry(payload: dict[str, Any], db: Session) -> dict:
    """Persist GitHub ``workflow_job`` lifecycle data and timing assessment.

    Job telemetry is never forwarded to ``CICDPredictor``. It is runtime/post-run
    operational data used only for bottleneck and stall analysis.
    """
    job_payload = payload.get("workflow_job") or {}
    provider_run_id = job_payload.get("run_id") or job_payload.get("workflow_run_id")
    provider_job_id = job_payload.get("id")
    if provider_run_id is None or provider_job_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="workflow_job id and run_id are required")
    pipeline = _find_or_create_pipeline(_pipeline_name_from_github(payload, "github-actions"), db)
    workflow = (
        db.query(WorkflowRun)
        .filter(
            WorkflowRun.provider == "github_actions",
            WorkflowRun.pipeline_id == pipeline.id,
            WorkflowRun.provider_run_id == str(provider_run_id),
        )
        .first()
    )
    if workflow is None:
        workflow = WorkflowRun(
            pipeline_id=pipeline.id,
            provider="github_actions",
            provider_run_id=str(provider_run_id),
        )
        db.add(workflow)
        db.flush()
    workflow.workflow_name = job_payload.get("workflow_name") or payload.get("workflow", {}).get("name") or workflow.workflow_name
    workflow.status = _merge_status(workflow.status, job_payload.get("status"))
    if job_payload.get("conclusion"):
        workflow.conclusion = job_payload.get("conclusion")
    workflow.branch = job_payload.get("head_branch") or workflow.branch
    workflow.head_sha = job_payload.get("head_sha") or workflow.head_sha
    workflow.queued_at = _earliest(workflow.queued_at, _parse_provider_time(job_payload.get("created_at")))
    workflow.started_at = _earliest(workflow.started_at, _parse_provider_time(job_payload.get("started_at")))
    workflow.completed_at = _latest(workflow.completed_at, _parse_provider_time(job_payload.get("completed_at")))
    workflow.raw_payload = payload

    job = (
        db.query(JobRun)
        .filter(JobRun.workflow_run_id == workflow.id, JobRun.provider_job_id == str(provider_job_id))
        .first()
    )
    if job is None:
        job = JobRun(workflow_run_id=workflow.id, provider_job_id=str(provider_job_id), name=str(job_payload.get("name") or "unnamed-job"))
        db.add(job)
        db.flush()
    job.name = str(job_payload.get("name") or job.name)
    job.status = _merge_status(job.status, job_payload.get("status"))
    if job_payload.get("conclusion"):
        job.conclusion = job_payload.get("conclusion")
    job.queued_at = _earliest(job.queued_at, _parse_provider_time(job_payload.get("created_at")))
    job.started_at = _earliest(job.started_at, _parse_provider_time(job_payload.get("started_at")))
    job.completed_at = _latest(job.completed_at, _parse_provider_time(job_payload.get("completed_at")))
    job.duration_seconds = _duration_seconds(job.started_at, job.completed_at)
    job.runner_labels = list(job_payload.get("labels") or [])
    job.raw_payload = job_payload
    _upsert_job_steps(job, list(job_payload.get("steps") or []), db)
    db.flush()

    history = [
        row[0]
        for row in (
            db.query(JobRun.duration_seconds)
            .join(WorkflowRun)
            .filter(
                WorkflowRun.pipeline_id == pipeline.id,
                JobRun.name == job.name,
                JobRun.id != job.id,
                JobRun.duration_seconds.isnot(None),
                JobRun.completed_at.isnot(None),
            )
            .all()
        )
    ]
    elapsed = job.duration_seconds
    in_progress = job.status.lower() in {"queued", "in_progress", "waiting"}
    if elapsed is None and in_progress and job.started_at:
        elapsed = _duration_seconds(job.started_at, datetime.now(timezone.utc))
    if elapsed is None:
        assessment = assess_job_timing(0.0, history)
    else:
        assessment = assess_job_timing(elapsed, history)
    job.baseline_sample_size = assessment.sample_size
    job.baseline_median_seconds = assessment.median_seconds
    job.baseline_p95_seconds = assessment.p95_seconds
    if elapsed is None and job.status.lower() in {"queued", "waiting"}:
        job.bottleneck_status = "awaiting_start"
    elif elapsed is None and job.status.lower() == "completed":
        job.bottleneck_status = "missing_timestamps"
    else:
        job.bottleneck_status = "stalled" if in_progress and assessment.status == "bottleneck" else assessment.status
    db.commit()
    return {
        "accepted": True,
        "telemetry_type": "workflow_job",
        "pipeline": pipeline.name,
        "workflow_run_id": workflow.id,
        "job_run_id": job.id,
        "provider_run_id": workflow.provider_run_id,
        "provider_job_id": job.provider_job_id,
        "job_status": job.status,
        "bottleneck_status": job.bottleneck_status,
        "baseline_sample_size": job.baseline_sample_size,
        "duration_seconds": job.duration_seconds,
    }


def _record_github_workflow_outcome(payload: dict[str, Any], db: Session) -> dict:
    """Persist a final workflow outcome without creating a post-build prediction."""
    run = payload.get("workflow_run") or {}
    provider_run_id = run.get("id")
    if provider_run_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="workflow_run id is required")
    conclusion = str(run.get("conclusion") or "").lower()
    if conclusion not in {"success", "failure", "cancelled", "timed_out", "action_required"}:
        return {"accepted": True, "ignored": True, "reason": "unsupported or missing workflow conclusion"}
    repository = run.get("head_repository") or payload.get("repository") or {}
    pipeline_name = repository if isinstance(repository, str) else repository.get("full_name") or repository.get("name") or str(provider_run_id)
    pipeline = _find_or_create_pipeline(pipeline_name, db)
    existing = (
        db.query(ProviderWorkflowOutcome)
        .filter(
            ProviderWorkflowOutcome.provider == "github_actions",
            ProviderWorkflowOutcome.pipeline_id == pipeline.id,
            ProviderWorkflowOutcome.provider_run_id == str(provider_run_id),
        )
        .first()
    )
    if existing is not None:
        return {"accepted": True, "duplicate": True, "workflow_outcome_id": existing.id, "pipeline": pipeline.name}
    outcome = ProviderWorkflowOutcome(
        pipeline_id=pipeline.id,
        provider="github_actions",
        provider_run_id=str(provider_run_id),
        workflow_name=run.get("name"),
        conclusion=conclusion,
        completed_at=_parse_provider_time(run.get("updated_at") or run.get("completed_at")),
        raw_payload=payload,
    )
    db.add(outcome)
    db.commit()
    return {"accepted": True, "duplicate": False, "workflow_outcome_id": outcome.id, "pipeline": pipeline.name}


@router.post("/github-actions")
async def github_actions_webhook(
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    """Ingest signed GitHub telemetry and final workflow outcome audit events."""
    body = await request.body()
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    signature = request.headers.get("X-Hub-Signature-256")
    if not secret or not verify_github_signature(body, signature, secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid GitHub webhook signature")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON webhook payload") from error
    event = request.headers.get("X-GitHub-Event", "")
    if event == "workflow_job" or (not event and "workflow_job" in payload):
        return _github_job_telemetry(payload, db)
    if event == "workflow_run" or (not event and "workflow_run" in payload):
        run = payload.get("workflow_run") or {}
        if payload.get("action") != "completed" or str(run.get("status") or "").lower() != "completed":
            return {"accepted": True, "ignored": True, "reason": "workflow run is not complete"}
        return _record_github_workflow_outcome(payload, db)
    return {"accepted": True, "ignored": True, "reason": "unsupported GitHub event"}


@router.post("/github-actions/job-log", response_model=JobFailureAnalysisOut)
def github_actions_job_log_collector(
    payload: GitHubJobLogCollectorIn,
    _secret: None = Depends(verify_collector_secret),
    db: Session = Depends(get_db),
) -> dict:
    """Accept an actual failed-job log fetched by a trusted Actions collector.

    GitHub job webhooks intentionally do not contain log output.  This endpoint
    keys the log by repository and GitHub provider job ID after the signed
    ``workflow_job`` delivery has created its telemetry row.
    """
    pipeline = db.query(Pipeline).filter(Pipeline.name == payload.repository).first()
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline telemetry not found")
    job = (
        db.query(JobRun)
        .join(WorkflowRun)
        .filter(
            WorkflowRun.pipeline_id == pipeline.id,
            WorkflowRun.provider == "github_actions",
            JobRun.provider_job_id == payload.provider_job_id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job telemetry not found")
    if str(job.conclusion or "").lower() not in {"failure", "timed_out", "cancelled", "action_required"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Failure logs may be attached only to a failed job")
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


@router.post("/github-actions/job-telemetry")
def github_actions_job_telemetry_collector(
    payload: GitHubJobTelemetryCollectorIn,
    _secret: None = Depends(verify_collector_secret),
    db: Session = Depends(get_db),
) -> dict:
    """Upsert queued/in-progress job state from a trusted scheduled collector.

    GitHub's ``workflow_job`` webhook fires at completion, so active-stall
    classification needs this separate Actions-read polling path.  The same
    idempotent provider job identity and telemetry persistence are reused; it
    does not call the pre-build predictor.
    """
    return _github_job_telemetry(
        {"repository": {"full_name": payload.repository}, "workflow_job": payload.workflow_job},
        db,
    )


@router.post("/jenkins")
def jenkins_webhook(
    payload: dict[str, Any],
    _secret: None = Depends(verify_webhook_secret),
    db: Session = Depends(get_db),
    predictor: CICDPredictor = Depends(get_predictor),
) -> dict:
    """Ingest a generic Jenkins post-build HTTP notification."""
    # Jenkins has no provider-standard HMAC signature, so its explicit shared header remains appropriate.
    result = str(payload.get("result", "")).upper()
    build = BuildFeaturesIn(
        build_features={
            "git_diff_src_churn": 0,
            "git_diff_src_files": 0,
            "git_diff_test_churn": 0,
            "git_diff_test_files": 0,
            "gh_lang": payload.get("language"),
            "git_branch": payload.get("branch", ""),
            "gh_build_started_at": payload.get("timestamp"),
            "jenkins_build_number": payload.get("build_number"),
            "jenkins_duration": payload.get("duration"),
        },
        build_status="failed" if result in {"FAILURE", "ABORTED", "UNSTABLE"} else "passed",
    )
    return create_build(
        pipeline_id=str(payload.get("job_name") or "jenkins"),
        payload=build,
        explain=False,
        db=db,
        predictor=predictor,
    )
