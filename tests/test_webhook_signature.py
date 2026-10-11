"""GitHub HMAC-SHA256 webhook signature verification tests."""

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth.security import get_password_hash
from api.db.database import Base, get_db
from api.db.models import User, UserRole
from api.main import app
from api.routers.pipelines import get_predictor


class StubPredictor:
    best_model_name = "xgboost"

    def full_analysis(self, _features, _log=None):
        return {"prediction": {"failure_probability": 0.6, "predicted_status": "failed", "confidence": 0.6}, "anomaly": {"is_anomaly": False, "anomaly_score": 0.2}, "root_cause": {"cluster_id": 0, "likely_cause": "test failure"}}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "signing-secret")
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine); factory = sessionmaker(bind=engine)
    session = factory(); session.add(User(username="admin", hashed_password=get_password_hash("password"), role=UserRole.ADMINISTRATOR)); session.commit()
    def override_db():
        db = factory()
        try: yield db
        finally: db.close()
    app.dependency_overrides[get_db] = override_db; app.dependency_overrides[get_predictor] = lambda: StubPredictor()
    with TestClient(app) as test_client: yield test_client
    app.dependency_overrides.clear(); session.close(); Base.metadata.drop_all(engine)


def _body():
    return json.dumps({"action":"completed","repository":{"name":"signed-repo"},"workflow_run":{"id":1,"name":"CI","status":"completed","conclusion":"failure","run_started_at":"2024-01-01T00:00:00Z","head_branch":"main","head_repository":{"name":"signed-repo"},"pull_requests":[]}}).encode()


def _signature(body):
    return "sha256=" + hmac.new(b"signing-secret", body, hashlib.sha256).hexdigest()


def test_github_signature_accepts_valid_body_and_rejects_tampering(client):
    body = _body()
    assert client.post("/webhooks/github-actions", content=body, headers={"X-Hub-Signature-256": _signature(body)}).status_code == 200
    assert client.post("/webhooks/github-actions", content=body.replace(b"signed-repo", b"tampered-repo"), headers={"X-Hub-Signature-256": _signature(body)}).status_code == 401
    assert client.post("/webhooks/github-actions", content=body).status_code == 401
    assert client.post("/webhooks/github-actions", content=body, headers={"X-Hub-Signature-256": "garbled"}).status_code == 401
