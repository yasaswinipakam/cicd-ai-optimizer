"""SQLAlchemy entities for CI/CD audit records and predictions."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Enum as SqlEnum, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from api.db.database import Base


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for ORM defaults."""
    return datetime.now(timezone.utc)


class UserRole(str, Enum):
    DEVELOPER = "developer"
    ADMINISTRATOR = "administrator"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SqlEnum(UserRole), nullable=False, default=UserRole.DEVELOPER)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    pipelines = relationship("Pipeline", back_populates="owner", cascade="all, delete-orphan")


class Pipeline(Base):
    __tablename__ = "pipelines"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="active")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    owner = relationship("User", back_populates="pipelines")
    build_logs = relationship("BuildLog", back_populates="pipeline", cascade="all, delete-orphan")
    workflow_runs = relationship("WorkflowRun", back_populates="pipeline", cascade="all, delete-orphan")
    provider_workflow_outcomes = relationship("ProviderWorkflowOutcome", back_populates="pipeline", cascade="all, delete-orphan")


class BuildLog(Base):
    __tablename__ = "build_logs"

    id = Column(Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    log_content = Column(Text, nullable=True)
    build_status = Column(String(50), nullable=False)
    stage_durations = Column(JSON, nullable=False, default=dict)
    raw_features = Column(JSON, nullable=False, default=dict)

    pipeline = relationship("Pipeline", back_populates="build_logs")
    predictions = relationship("Prediction", back_populates="build_log", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    build_log_id = Column(Integer, ForeignKey("build_logs.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    failure_probability = Column(Float, nullable=False)
    predicted_status = Column(String(50), nullable=False)
    is_anomaly = Column(Boolean, nullable=False, default=False)
    anomaly_score = Column(Float, nullable=True)
    root_cause_cluster_id = Column(Integer, nullable=True)
    root_cause_label = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    build_log = relationship("BuildLog", back_populates="predictions")


class WorkflowRun(Base):
    """A provider workflow execution used exclusively for runtime telemetry.

    This is intentionally distinct from ``BuildLog``/``Prediction``: workflow
    status, completion times, and job durations arrive during or after execution
    and must never become pre-build model features.
    """

    __tablename__ = "workflow_runs"
    __table_args__ = (UniqueConstraint("provider", "pipeline_id", "provider_run_id", name="uq_workflow_run_provider_id"),)

    id = Column(Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False, index=True)
    provider = Column(String(50), nullable=False, default="github_actions")
    provider_run_id = Column(String(100), nullable=False)
    workflow_name = Column(String(500), nullable=True)
    status = Column(String(50), nullable=False, default="queued")
    conclusion = Column(String(100), nullable=True)
    branch = Column(String(255), nullable=True)
    head_sha = Column(String(100), nullable=True)
    queued_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    raw_payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    pipeline = relationship("Pipeline", back_populates="workflow_runs")
    jobs = relationship("JobRun", back_populates="workflow_run", cascade="all, delete-orphan")


class ProviderWorkflowOutcome(Base):
    """Idempotent final-state audit record for a provider workflow run.

    A completed workflow is post-execution evidence, so it is deliberately not
    a ``BuildLog`` or ``Prediction``.  This prevents a completion webhook from
    masquerading as a pre-build inference request.
    """

    __tablename__ = "provider_workflow_outcomes"
    __table_args__ = (UniqueConstraint("provider", "pipeline_id", "provider_run_id", name="uq_provider_workflow_outcome"),)

    id = Column(Integer, primary_key=True)
    pipeline_id = Column(Integer, ForeignKey("pipelines.id"), nullable=False, index=True)
    provider = Column(String(50), nullable=False, default="github_actions")
    provider_run_id = Column(String(100), nullable=False)
    workflow_name = Column(String(500), nullable=True)
    conclusion = Column(String(100), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    raw_payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    pipeline = relationship("Pipeline", back_populates="provider_workflow_outcomes")


class JobRun(Base):
    """One CI job with timing and derived bottleneck/stall assessment."""

    __tablename__ = "job_runs"
    __table_args__ = (UniqueConstraint("workflow_run_id", "provider_job_id", name="uq_job_run_provider_id"),)

    id = Column(Integer, primary_key=True)
    workflow_run_id = Column(Integer, ForeignKey("workflow_runs.id"), nullable=False, index=True)
    provider_job_id = Column(String(100), nullable=False)
    name = Column(String(500), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="queued")
    conclusion = Column(String(100), nullable=True)
    queued_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    runner_labels = Column(JSON, nullable=False, default=list)
    baseline_sample_size = Column(Integer, nullable=False, default=0)
    baseline_median_seconds = Column(Float, nullable=True)
    baseline_p95_seconds = Column(Float, nullable=True)
    bottleneck_status = Column(String(50), nullable=False, default="insufficient_history")
    raw_payload = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    workflow_run = relationship("WorkflowRun", back_populates="jobs")
    steps = relationship("JobStepRun", back_populates="job_run", cascade="all, delete-orphan")
    failure_analysis = relationship("JobFailureAnalysis", back_populates="job_run", cascade="all, delete-orphan", uselist=False)


class JobStepRun(Base):
    """A GitHub Actions job step, retained for post-run bottleneck diagnostics."""

    __tablename__ = "job_step_runs"
    __table_args__ = (UniqueConstraint("job_run_id", "provider_step_number", name="uq_job_step_number"),)

    id = Column(Integer, primary_key=True)
    job_run_id = Column(Integer, ForeignKey("job_runs.id"), nullable=False, index=True)
    provider_step_number = Column(Integer, nullable=False)
    name = Column(String(500), nullable=False)
    status = Column(String(50), nullable=False, default="queued")
    conclusion = Column(String(100), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    raw_payload = Column(JSON, nullable=False, default=dict)

    job_run = relationship("JobRun", back_populates="steps")


class JobFailureAnalysis(Base):
    """Sanitized, post-failure log diagnosis for a GitHub Actions job.

    This deliberately belongs to runtime diagnostics rather than BuildLog or
    Prediction, preventing post-build evidence from entering pre-build scoring.
    """

    __tablename__ = "job_failure_analyses"

    id = Column(Integer, primary_key=True)
    job_run_id = Column(Integer, ForeignKey("job_runs.id"), nullable=False, unique=True, index=True)
    log_source = Column(String(100), nullable=False, default="submitted")
    sanitized_log = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    summary = Column(Text, nullable=False)
    signature = Column(String(100), nullable=False, index=True)
    matched_signals = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    job_run = relationship("JobRun", back_populates="failure_analysis")
