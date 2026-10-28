"""GitHub Actions Tier E deployment: training-equivalent live context, inputs, and webhook scoring."""

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.db.database import Base, get_db
from api.db.models import BuildLog
from api.main import app
from api.routers.pipelines import get_predictor
from src.features.schema import FEATURE_COLS
from src.features.stream_context import CONTEXT_COLS, PriorBuild, gha_stream_key, stream_context
from src.inference.predictor import CICDPredictor
from src.models.anomaly import AnomalyDetector
from src.models.clustering import RootCauseClusterer
from src.models.tier_e import TierEBlend
from src.models.train_tier_e_gha import gha

NUMERIC, CATEGORICAL = gha.FEATURES["E"]


def _raw_runs(path, n: int = 300, seed: int = 5):
    """A CSV in the GHA dataset format: overlapping runs, PRs from two authors, two workflows."""
    rng = np.random.default_rng(seed)
    created = pd.Timestamp("2026-08-15", tz="UTC") + pd.to_timedelta(np.sort(rng.integers(0, 6 * 24 * 60, n)), unit="m")
    event = rng.choice(["push", "pull_request"], n, p=[0.6, 0.4])
    runs = pd.DataFrame({
        "run_id": np.arange(5000, 5000 + n), "repo": "acme/api", "workflow_id": rng.choice([11, 12], n),
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at": (created + pd.to_timedelta(rng.integers(1, 8 * 60, n), unit="m")).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conclusion": rng.choice(["success", "failure"], n, p=[0.7, 0.3]), "run_attempt": 1, "event": event,
        "head_branch": rng.choice(["main", "feat-a", "feat-b"], n), "actor_login": rng.choice(["alice", "bob"], n),
        "head_sha": [f"sha{i}" for i in range(n)], "duration": 60.0, "additions": 10, "deletions": 2,
        "total_churn": 12, "files_modified": 3, "msg_len": 40, "is_merge": False, "num_parents": 1,
        "time_since_last_commit": 600.0,
    })
    runs.to_csv(path, index=False)
    return path


def test_live_gha_context_matches_the_training_pass(tmp_path) -> None:
    gha.configure("gha_test", [_raw_runs(tmp_path / "runs.csv")], gha.PROTOCOL)
    batch = gha.with_history(gha.load_runs("primary")[0], 0)
    prior: list[PriorBuild] = []
    for i, row in batch.iterrows():
        event = "pull_request" if row.gh_is_pr else "push"
        actor = row.gh_pull_req_num.split(":")[0] if row.gh_is_pr else None
        stream = gha_stream_key(event, actor, row.git_branch)
        assert stream == row.stream
        live = stream_context(prior, started_at=row.gh_build_started_at, stream=stream,
                              previous_build_id=None if pd.isna(row.tr_prev_build) else int(row.tr_prev_build))
        for column in CONTEXT_COLS:
            expected = batch.at[i, column]
            assert (pd.isna(expected) and pd.isna(live[column])) or np.isclose(live[column], expected), (i, column)
        prior.append(PriorBuild(int(row.tr_build_id), row.gh_build_started_at, stream, row._finished_at,
                                bool(row.target_failed)))
    assert batch.ctx_prevbuild_pending.eq(1).any() and batch.hist_prevbuild_failed.notna().any()
    gha.configure("gha_tier_e", [gha.GHA_RAW], gha.PROTOCOL)


def _training_frame(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.random((n, len(NUMERIC))), columns=NUMERIC)
    d["hist_stream_prev_failed"] = rng.integers(0, 2, n).astype(float)
    d["target_failed"] = (rng.random(n) < np.where(d.hist_stream_prev_failed.eq(1), 0.7, 0.1)).astype(int)
    d["workflow"] = rng.choice(["11", "12"], n)
    d["gh_project_name"] = rng.choice(["acme/api", "acme/web"], n)
    d["gh_build_started_at"] = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
    return d


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    path = tmp_path_factory.mktemp("tier_e_gha_artifacts")
    model = TierEBlend(NUMERIC, CATEGORICAL).fit(_training_frame())
    model.metadata = {"variant": "E_lgbmblend", "input_schema": "github_actions", "decision_threshold": 0.5}
    model.input_fill = {"additions": 30.0, "deletions": 5.0, "files_modified": 2.0}
    joblib.dump(model, path / "tier_e_gha_blend.joblib")
    matrix = pd.DataFrame([[0.0] * len(FEATURE_COLS), [1.0] * len(FEATURE_COLS)], columns=FEATURE_COLS)
    imputer = SimpleImputer(strategy="median").fit(matrix)
    joblib.dump({"imputer": imputer, "scaler": StandardScaler().fit(matrix), "feature_cols": FEATURE_COLS.copy(),
                 "cold_start_prior": 0.2, "decision_threshold": 0.9, "training_mode": "full"}, path / "preprocessing.joblib")
    joblib.dump({"prev_status_enc": LabelEncoder().fit(["__missing__"]), "lang_enc": LabelEncoder().fit(["__missing__"])},
                path / "encoders.joblib")
    AnomalyDetector(contamination=0.5).fit(imputer.transform(matrix)).save(str(path / "anomaly_detector.joblib"))
    RootCauseClusterer(n_clusters=2).fit(pd.Series(["database timeout", "database refused"])).save(
        str(path / "root_cause_clusterer.joblib"))
    (path / "best_model.txt").write_text("tier_e_gha_blend\n", encoding="utf-8")
    return path


def test_predictor_derives_github_actions_inputs(artifacts) -> None:
    predictor = CICDPredictor(str(artifacts))
    assert predictor.input_schema == "github_actions" and predictor.uses_stream_context
    assert predictor.preprocessing["training_mode"] == "tier_e_gha:E_lgbmblend"
    frame = predictor._tier_e_frame({"event": "pull_request", "head_branch": "main", "workflow_id": 11,
                                     "additions": 7, "deletions": 3, "num_parents": 2, "commit_message": "fix"})
    row = frame.iloc[0]
    assert (row.is_pr, row.is_default_branch, row.total_churn, row.is_merge, row.msg_len) == (1, 1, 10, 1, 3)
    assert row.workflow == "11" and row.files_modified == 2.0  # omitted: training median
    push = predictor._tier_e_frame({"event": "push", "head_branch": "feat"}).iloc[0]
    assert (push.is_pr, push.is_default_branch, push.additions) == (0, 0, 30.0)
    assert np.isnan(push.hist_stream_completed)  # cold start stays missing


@pytest.fixture()
def client(artifacts, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    predictor = CICDPredictor(str(artifacts))
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_predictor] = lambda: predictor
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)


def _send(client: TestClient, action: str, **run) -> dict:
    run = {"workflow_id": 11, "name": "CI", "event": "push", "head_branch": "main", "run_attempt": 1,
           "actor": {"login": "alice"}, "status": "completed" if action == "completed" else "queued", **run}
    body = json.dumps({"action": action, "repository": {"full_name": "acme/api"}, "workflow_run": run}).encode()
    signature = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
    response = client.post("/webhooks/github-actions", content=body,
                           headers={"X-Hub-Signature-256": signature, "X-GitHub-Event": "workflow_run"})
    assert response.status_code == 200, response.text
    return response.json()


def _context(client: TestClient, build_id: int) -> dict:
    db = next(app.dependency_overrides[get_db]())
    try:
        return db.get(BuildLog, build_id).raw_features["_stream_context"]
    finally:
        db.close()


def test_requested_runs_are_scored_and_completed_runs_become_history(client: TestClient) -> None:
    now = datetime.now(timezone.utc)
    commit = {"timestamp": (now - timedelta(hours=1)).isoformat(), "message": "first"}
    first = _send(client, "requested", id=1, head_sha="a1", head_commit=commit)
    assert first["duplicate"] is False and 0 < first["failure_probability"] < 1
    assert _send(client, "requested", id=1, head_sha="a1")["duplicate"] is True
    assert _send(client, "requested", id=9, run_attempt=2)["ignored"] is True
    assert _send(client, "requested", id=8, event="schedule")["ignored"] is True

    second = _send(client, "requested", id=2, head_sha="b2",
                   head_commit={"timestamp": now.isoformat(), "message": "second"})
    running = _context(client, second["build_log_id"])
    assert running["ctx_stream_pending"] == 1 and running["ctx_prevbuild_pending"] == 1
    assert running["time_since_last_commit"] == pytest.approx(3600)

    done = _send(client, "completed", id=1, conclusion="failure", updated_at=now.isoformat())
    assert done["shadow_build_id"] == first["build_log_id"]
    _send(client, "completed", id=2, conclusion="cancelled", updated_at=now.isoformat())
    pr = _send(client, "requested", id=3, event="pull_request", head_branch="feat", actor={"login": "bob"})
    third = _send(client, "requested", id=4, head_sha="c3")
    context = _context(client, third["build_log_id"])
    # Run 1's failure is history.  Cancelled run 2 is neither history nor in flight,
    # and, as in training (which has no cancelled runs), not the previous run either.
    assert context["hist_stream_completed"] == 1 and context["hist_stream_prev_failed"] == 1
    assert context["hist_prevbuild_failed"] == 1 and context["ctx_prevbuild_pending"] == 0
    assert context["ctx_repo_pending"] == 1  # only bob's PR run
    assert _context(client, pr["build_log_id"])["hist_stream_completed"] == 0
    builds = {b["id"]: b["build_status"] for b in client.get("/pipelines/acme/api/builds").json()}
    assert builds[first["build_log_id"]] == "failed" and builds[second["build_log_id"]] == "cancelled"
    assert client.get("/model").json()["input_schema"] == "github_actions"


def test_runs_are_timed_by_githubs_creation_time_not_delivery(client: TestClient) -> None:
    start = datetime.now(timezone.utc) - timedelta(days=2)
    at = lambda hours: (start + timedelta(hours=hours)).isoformat()
    first = _send(client, "requested", id=21, head_sha="d1", created_at=at(0))
    _send(client, "completed", id=21, conclusion="failure", created_at=at(0), updated_at=at(0.5))
    second = _context(client, _send(client, "requested", id=22, head_sha="d2", created_at=at(2))["build_log_id"])
    # Two hours apart by GitHub's clock, although delivered moments apart.
    assert second["ctx_hours_since_stream_start"] == pytest.approx(2) and second["ctx_repo_started_1h"] == 0
    assert second["hist_stream_prev_failed"] == 1
    # Delivered last but created first: runs that started after it are not its history or in flight.
    late = _context(client, _send(client, "requested", id=23, head_sha="d3", created_at=at(-1))["build_log_id"])
    assert late["hist_stream_completed"] == 0 and late["ctx_repo_pending"] == 0
    builds = {b["id"]: b["timestamp"] for b in client.get("/pipelines/acme/api/builds").json()}
    assert builds[first["build_log_id"]].startswith(start.strftime("%Y-%m-%dT%H"))
