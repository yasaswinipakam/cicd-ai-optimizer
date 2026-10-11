"""Shared persistence for strictly post-failure CI diagnostics."""

from __future__ import annotations

from sqlalchemy.orm import Session

from api.db.models import JobFailureAnalysis, JobRun
from src.diagnostics.log_analysis import analyze_failed_log


def save_job_failure_analysis(
    job: JobRun, log_content: str, log_source: str, db: Session
) -> JobFailureAnalysis:
    """Sanitize/classify a real log and upsert one analysis per job run."""
    result = analyze_failed_log(log_content)
    analysis = job.failure_analysis
    if analysis is None:
        analysis = JobFailureAnalysis(job_run_id=job.id)
        db.add(analysis)
    analysis.log_source = log_source
    analysis.sanitized_log = result.sanitized_log
    analysis.category = result.category
    analysis.title = result.title
    analysis.summary = result.summary
    analysis.signature = result.signature
    analysis.matched_signals = result.matched_signals
    db.commit()
    db.refresh(analysis)
    return analysis
