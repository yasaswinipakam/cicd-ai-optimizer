"""Webhook signature and GitHub Actions ingestion tests."""

import pytest
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.db.database import Base, get_db
from api.db.models import BuildLog, JobFailureAnalysis, JobRun, JobStepRun, Pipeline, ProviderWorkflowOutcome, WorkflowRun
from api.main import app
from api.routers.pipelines import get_predictor


class StubPredictor:
    best_model_name = "xgboost"

    def full_analysis(self, _features, _log=None):
        return {"prediction": {"failure_probability": 0.6, "predicted_status": "failed", "confidence": 0.6}, "anomaly": {"is_anomaly": False, "anomaly_score": 0.2}, "root_cause": {"cluster_id": 0, "likely_cause": "test failure"}}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("CI_DIAGNOSTICS_COLLECTOR_SECRET", "collector-secret")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    def override_db():
        db = factory()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_predictor] = lambda: StubPredictor()
    with TestClient(app) as test_client: yield test_client
    app.dependency_overrides.clear(); session.close(); Base.metadata.drop_all(engine)


def test_completed_workflow_run_is_a_idempotent_outcome_audit_not_a_prediction(client: TestClient):
    payload = {"action": "completed", "repository": {"name": "demo-repo"}, "workflow_run": {"id": 5, "name": "CI", "status": "completed", "conclusion": "failure", "run_started_at": "2024-01-01T12:00:00Z", "head_branch": "main", "head_repository": {"name": "demo-repo"}, "pull_requests": []}}
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert client.post("/webhooks/github-actions", content=body).status_code == 401
    response = client.post("/webhooks/github-actions", content=body, headers={"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_run"})
    assert response.status_code == 200
    assert response.json()["duplicate"] is False
    repeated = client.post("/webhooks/github-actions", content=body, headers={"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_run"})
    assert repeated.json()["duplicate"] is True
    db = next(app.dependency_overrides[get_db]())
    try:
        assert db.query(ProviderWorkflowOutcome).count() == 1
        assert db.query(BuildLog).count() == 0
    finally:
        db.close()
    summary = client.get("/dashboard/summary")
    assert summary.status_code == 200
    assert summary.json()["total_builds_analyzed"] == 0


def test_incomplete_or_unrelated_signed_events_do_not_create_outcomes_or_predictions(client: TestClient):
    payload = {"action": "in_progress", "repository": {"name": "demo-repo"}, "workflow_run": {"id": 6, "status": "in_progress"}}
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    response = client.post("/webhooks/github-actions", content=body, headers={"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_run"})
    assert response.status_code == 200 and response.json()["ignored"] is True
    ping = json.dumps({"zen": "test"}).encode()
    ping_signature = "sha256=" + hmac.new(b"test-secret", ping, hashlib.sha256).hexdigest()
    assert client.post("/webhooks/github-actions", content=ping, headers={"X-Hub-Signature-256": ping_signature, "X-GitHub-Event": "ping"}).json()["ignored"] is True
    db = next(app.dependency_overrides[get_db]())
    try:
        assert db.query(ProviderWorkflowOutcome).count() == 0
        assert db.query(BuildLog).count() == 0
    finally:
        db.close()


def test_github_workflow_job_webhook_persists_runtime_telemetry_without_prediction(client: TestClient):
    payload = {
        "action": "completed",
        "repository": {"full_name": "acme/demo"},
        "workflow_job": {
            "id": 9001,
            "run_id": 7001,
            "workflow_name": "CI",
            "name": "unit tests",
            "status": "completed",
            "conclusion": "success",
            "created_at": "2026-01-01T10:00:00Z",
            "started_at": "2026-01-01T10:00:10Z",
            "completed_at": "2026-01-01T10:02:10Z",
            "head_branch": "main",
            "head_sha": "abc123",
            "labels": ["ubuntu-latest"],
            "steps": [
                {"number": 1, "name": "Checkout", "status": "completed", "conclusion": "success", "started_at": "2026-01-01T10:00:10Z", "completed_at": "2026-01-01T10:00:20Z"},
                {"number": 2, "name": "Test", "status": "completed", "conclusion": "success", "started_at": "2026-01-01T10:00:20Z", "completed_at": "2026-01-01T10:02:10Z"},
            ],
        },
    }
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    headers = {"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_job"}
    response = client.post("/webhooks/github-actions", content=body, headers=headers)
    assert response.status_code == 200
    assert response.json()["telemetry_type"] == "workflow_job"
    assert response.json()["duration_seconds"] == 120.0
    # Repeat delivery is a normal GitHub webhook possibility: keep it idempotent.
    assert client.post("/webhooks/github-actions", content=body, headers=headers).status_code == 200

    db = next(app.dependency_overrides[get_db]())
    try:
        assert db.query(WorkflowRun).count() == 1
        assert db.query(JobRun).count() == 1
        assert db.query(JobStepRun).count() == 2
        assert db.query(BuildLog).count() == 0
        job = db.query(JobRun).one()
        assert job.bottleneck_status == "insufficient_history"
        assert job.runner_labels == ["ubuntu-latest"]
    finally:
        db.close()

    telemetry = client.get("/pipelines/acme/demo/telemetry/jobs")
    assert telemetry.status_code == 200
    assert telemetry.json()[0]["job_name"] == "unit tests"
    assert telemetry.json()[0]["bottleneck_status"] == "insufficient_history"
    operational = client.get("/dashboard/operational-summary")
    assert operational.status_code == 200
    assert operational.json()["total_jobs_observed"] == 1
    assert operational.json()["failed_jobs_with_captured_logs"] == 0


def test_late_queued_delivery_does_not_regress_completed_job_and_missing_times_are_explicit(client: TestClient):
    complete = {
        "repository": {"full_name": "acme/order"},
        "workflow_job": {"id": 9010, "run_id": 7010, "name": "test", "status": "completed", "conclusion": "success", "created_at": "2026-01-01T10:00:00Z", "started_at": "2026-01-01T10:00:10Z", "completed_at": "2026-01-01T10:02:10Z", "steps": []},
    }
    body = json.dumps(complete).encode(); signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    headers = {"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_job"}
    assert client.post("/webhooks/github-actions", content=body, headers=headers).status_code == 200
    delayed = {"repository": {"full_name": "acme/order"}, "workflow_job": {"id": 9010, "run_id": 7010, "name": "test", "status": "queued", "created_at": "2026-01-01T10:00:00Z", "steps": []}}
    delayed_body = json.dumps(delayed).encode(); delayed_signature = "sha256=" + hmac.new(b"test-secret", delayed_body, hashlib.sha256).hexdigest()
    assert client.post("/webhooks/github-actions", content=delayed_body, headers={"X-Hub-Signature-256": delayed_signature, "X-GitHub-Event": "workflow_job"}).status_code == 200
    missing_times = {"repository": {"full_name": "acme/order"}, "workflow_job": {"id": 9011, "run_id": 7011, "name": "lint", "status": "completed", "conclusion": "failure", "steps": []}}
    missing_body = json.dumps(missing_times).encode(); missing_signature = "sha256=" + hmac.new(b"test-secret", missing_body, hashlib.sha256).hexdigest()
    assert client.post("/webhooks/github-actions", content=missing_body, headers={"X-Hub-Signature-256": missing_signature, "X-GitHub-Event": "workflow_job"}).status_code == 200
    db = next(app.dependency_overrides[get_db]())
    try:
        job = db.query(JobRun).filter(JobRun.provider_job_id == "9010").one()
        assert job.status == "completed" and job.duration_seconds == 120.0
        assert db.query(JobRun).filter(JobRun.provider_job_id == "9011").one().bottleneck_status == "missing_timestamps"
    finally:
        db.close()


def test_failed_job_log_is_sanitized_and_classified_without_creating_prediction(client: TestClient):
    payload = {
        "action": "completed",
        "repository": {"full_name": "acme/demo"},
        "workflow_job": {
            "id": 9002, "run_id": 7002, "workflow_name": "CI", "name": "integration tests",
            "status": "completed", "conclusion": "failure",
            "created_at": "2026-01-01T10:00:00Z", "started_at": "2026-01-01T10:00:10Z",
            "completed_at": "2026-01-01T10:02:10Z", "steps": [],
        },
    }
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    response = client.post(
        "/webhooks/github-actions", content=body,
        headers={"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_job"},
    )
    assert response.status_code == 200
    job_id = response.json()["job_run_id"]
    log = "Request timed out while calling service\nTOKEN=super-secret-value\nError: connection reset"
    result = client.post(
        f"/pipelines/acme/demo/telemetry/jobs/{job_id}/failure-log",
        json={"log_content": log},
    )
    assert result.status_code == 200
    assert result.json()["category"] == "flaky_timing_network"
    assert "timed out" in result.json()["matched_signals"][0].lower()
    analysis = client.get(
        f"/pipelines/acme/demo/telemetry/jobs/{job_id}/failure-analysis",
    )
    assert analysis.status_code == 200
    db = next(app.dependency_overrides[get_db]())
    try:
        stored = db.query(JobFailureAnalysis).one()
        assert "super-secret-value" not in stored.sanitized_log
        assert db.query(BuildLog).count() == 0
    finally:
        db.close()


def test_collector_can_attach_real_log_by_provider_job_id_without_a_human_jwt(client: TestClient):
    payload = {
        "repository": {"full_name": "acme/collector-demo"},
        "workflow_job": {
            "id": 9003, "run_id": 7003, "name": "build", "status": "completed", "conclusion": "failure",
            "created_at": "2026-01-01T10:00:00Z", "started_at": "2026-01-01T10:00:10Z",
            "completed_at": "2026-01-01T10:02:10Z", "steps": [],
        },
    }
    body = json.dumps(payload).encode()
    signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    assert client.post(
        "/webhooks/github-actions", content=body,
        headers={"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_job"},
    ).status_code == 200
    collector_payload = {
        "repository": "acme/collector-demo", "provider_job_id": "9003",
        "log_content": "SyntaxError: unexpected token at line 42",
    }
    assert client.post("/webhooks/github-actions/job-log", json=collector_payload).status_code == 401
    response = client.post(
        "/webhooks/github-actions/job-log", json=collector_payload,
        headers={"X-CI-Diagnostics-Secret": "collector-secret"},
    )
    assert response.status_code == 200
    assert response.json()["category"] == "compilation_syntax"
    summary = client.get("/dashboard/operational-summary")
    assert summary.status_code == 200
    assert summary.json()["total_jobs_observed"] == 1
    assert summary.json()["failed_jobs_with_captured_logs"] == 1
    assert summary.json()["top_failure_categories"][0]["category"] == "compilation_syntax"


def test_active_job_poller_handles_queued_and_stalled_jobs_without_predictions(client: TestClient):
    def post_snapshot(job: dict, repository="acme/poller"):
        return client.post(
            "/webhooks/github-actions/job-telemetry",
            json={"repository": repository, "workflow_job": job},
            headers={"X-CI-Diagnostics-Secret": "collector-secret"},
        )

    queued = {"id": 9100, "run_id": 7100, "name": "deploy", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
    assert post_snapshot(queued).status_code == 200
    assert post_snapshot(queued).json()["bottleneck_status"] == "awaiting_start"
    # Build the same-name completed history that the active job will be compared against.
    for index in range(10):
        job = {
            "id": 9200 + index, "run_id": 7200 + index, "name": "deploy", "status": "completed", "conclusion": "success",
            "created_at": "2026-01-01T00:00:00Z", "started_at": "2026-01-01T00:00:00Z", "completed_at": "2026-01-01T00:02:00Z",
        }
        assert post_snapshot(job).status_code == 200
    started = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    active = {"id": 9300, "run_id": 7300, "name": "deploy", "status": "in_progress", "created_at": started, "started_at": started}
    response = post_snapshot(active)
    assert response.status_code == 200
    assert response.json()["bottleneck_status"] == "stalled"
    db = next(app.dependency_overrides[get_db]())
    try:
        assert db.query(BuildLog).count() == 0
        assert db.query(JobRun).count() == 12
    finally:
        db.close()
