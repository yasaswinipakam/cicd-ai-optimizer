"""Pydantic request and response contracts for the CI/CD optimizer API."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BuildFeaturesIn(BaseModel):
    """Incoming build data, accepted either flat or under ``build_features``."""

    model_config = ConfigDict(extra="allow")

    build_features: dict[str, Any] = Field(default_factory=dict)
    failed_test_log: str | None = None
    build_status: str | None = None
    stage_durations: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def collect_flat_features(cls, values: Any) -> Any:
        if not isinstance(values, dict) or "build_features" in values:
            return values
        reserved = {"failed_test_log", "build_status", "stage_durations"}
        normalized = {key: value for key, value in values.items() if key in reserved}
        normalized["build_features"] = {key: value for key, value in values.items() if key not in reserved}
        return normalized


ACTIONABLE_INTERVENTION_FIELDS = {
    "git_diff_src_churn": 2000.0,
    "git_diff_test_churn": 2000.0,
    "gh_diff_files_added": 500.0,
    "gh_diff_files_deleted": 500.0,
    "gh_diff_files_modified": 500.0,
}


class OptimizationInterventionIn(BaseModel):
    """One bounded, developer-plausible hypothetical raw-input change."""

    model_config = ConfigDict(extra="forbid")

    field: Literal[
        "git_diff_src_churn",
        "git_diff_test_churn",
        "gh_diff_files_added",
        "gh_diff_files_deleted",
        "gh_diff_files_modified",
    ]
    value: float
    reason: str | None = None

    @model_validator(mode="after")
    def validate_bounded_value(self) -> "OptimizationInterventionIn":
        maximum = ACTIONABLE_INTERVENTION_FIELDS[self.field]
        if not math.isfinite(self.value) or self.value < 0 or self.value > maximum:
            raise ValueError(f"{self.field} must be a finite value between 0 and {maximum:g}.")
        return self


class OptimizationSimulationIn(BaseModel):
    """Database-free request for exactly one model-based build-risk scenario."""

    model_config = ConfigDict(extra="forbid")

    build_features: dict[str, Any] = Field(default_factory=dict)
    intervention: OptimizationInterventionIn
    include_explanation: bool = False

    @model_validator(mode="after")
    def reject_post_build_inputs(self) -> "OptimizationSimulationIn":
        forbidden = [key for key in self.build_features if str(key).startswith("tr_log_")]
        if forbidden:
            raise ValueError("Post-build tr_log_* fields are not accepted for optimization simulation.")
        return self


class OptimizationScenarioOut(BaseModel):
    failure_probability: float
    predicted_status: str
    derived_features: dict[str, float]
    explanation: list[dict] | None = None


class OptimizationSimulationOut(BaseModel):
    simulation_type: Literal["model_based_hypothetical"]
    baseline: OptimizationScenarioOut
    simulation: OptimizationScenarioOut
    risk_delta: float
    percentage_point_change: float
    intervention: dict[str, Any]
    disclaimer: str


class PredictionOut(BaseModel):
    failure_probability: float
    predicted_status: str
    confidence: float
    is_anomaly: bool = Field(
        description="Whether the available pre-build input profile is unusual; this is not a failure prediction."
    )
    anomaly_score: float = Field(
        description="Isolation Forest decision score; lower values are more unusual input profiles."
    )
    root_cause_cluster_id: int | None = None
    root_cause_label: str | None = None
    explanation: list[dict] | None = None


class BuildCreatedOut(PredictionOut):
    build_log_id: int
    pipeline_id: int
    pipeline_name: str
    # Tier E history/in-flight values the server derived for this build;
    # None when the deployed model does not use them. Missing history is null.
    stream_context: dict[str, float | None] | None = None


class ShadowBuildCreatedOut(PredictionOut):
    """Non-blocking prediction record awaiting a later observed CI outcome."""

    build_log_id: int
    pipeline_id: int
    pipeline_name: str
    stream_context: dict[str, float | None] | None = None
    decision_threshold: float
    is_alert: bool
    model_version: str
    training_mode: str | None = None
    calibration_method: str = "raw"
    calibration_version: str | None = None


class ShadowOutcomeIn(BaseModel):
    """Observed result attached to a previously persisted shadow prediction."""

    build_status: Literal["passed", "failed"]
    failed_test_log: str | None = None
    stage_durations: dict[str, float] = Field(default_factory=dict)


class JobTelemetryOut(BaseModel):
    """Runtime-only GitHub/Jenkins job telemetry; never a prediction feature."""

    job_run_id: int
    workflow_run_id: int
    workflow_name: str | None = None
    job_name: str
    status: str
    conclusion: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    bottleneck_status: str
    baseline_sample_size: int
    baseline_median_seconds: float | None = None
    baseline_p95_seconds: float | None = None


class JobFailureLogIn(BaseModel):
    """A failed job's real console output, supplied after CI execution."""

    log_content: str = Field(min_length=1, max_length=200_000)
    log_source: str = Field(default="github_actions_collector", min_length=1, max_length=100)


class GitHubJobLogCollectorIn(JobFailureLogIn):
    """Collector payload keyed by public GitHub repository/job identifiers."""

    repository: str = Field(min_length=1, max_length=500)
    provider_job_id: str = Field(min_length=1, max_length=100)


class GitHubJobTelemetryCollectorIn(BaseModel):
    """Authenticated polling snapshot for one GitHub Actions job lifecycle."""

    repository: str = Field(min_length=1, max_length=500)
    workflow_job: dict[str, Any]


class JobFailureAnalysisOut(BaseModel):
    job_run_id: int
    category: str
    title: str
    summary: str
    signature: str
    matched_signals: list[str]
    created_at: datetime


class BuildLogOut(BaseModel):
    id: int
    timestamp: datetime
    build_status: str
    failure_probability: float | None = None
    predicted_status: str | None = None
    is_anomaly: bool | None = Field(
        default=None,
        description="Whether the stored pre-build input profile was unusual; this is not a failure prediction.",
    )
    anomaly_score: float | None = None
    root_cause_label: str | None = None
    optimization_suggestion: dict | None = None
    explanation: list[dict] | None = None


class PipelineOut(BaseModel):
    id: int
    name: str
    status: str
    latest_build_status: str | None = None
    latest_failure_probability: float | None = None


class RootCauseSummaryOut(BaseModel):
    cluster_id: int
    label: str | None = None
    count: int


class DashboardSummaryOut(BaseModel):
    total_builds_analyzed: int
    pipelines_monitored: int
    # Failure rate over builds with a reported outcome (builds_with_outcome of them).
    overall_failure_rate: float
    builds_with_outcome: int = 0
    anomalies_detected: int
    top_root_causes: list[RootCauseSummaryOut]


class OperationalJobTypeOut(BaseModel):
    job_name: str
    count: int
    bottleneck_or_stall_count: int
    average_duration_seconds: float | None = None


class OperationalFailureCategoryOut(BaseModel):
    category: str
    title: str
    count: int


class OperationalDashboardOut(BaseModel):
    """Post-execution telemetry aggregates, separate from ML prediction metrics."""

    total_jobs_observed: int
    bottlenecks_detected: int
    stalled_jobs_detected: int
    failed_jobs_with_captured_logs: int
    top_slowest_or_bottlenecked_jobs: list[OperationalJobTypeOut]
    top_failure_categories: list[OperationalFailureCategoryOut]


class FailureTrendOut(BaseModel):
    date: str
    total_builds: int
    failed_builds: int
    failure_rate: float
