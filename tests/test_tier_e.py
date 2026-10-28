"""Tier E deployment: fitted blend, predictor path, and server-derived stream context."""

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

from api.auth.security import get_password_hash
from api.db.database import Base, get_db
from api.db.models import User, UserRole
from api.main import app
from api.routers.pipelines import get_predictor
from src.features.schema import FEATURE_COLS
from src.inference.predictor import CICDPredictor
from src.models.anomaly import AnomalyDetector
from src.models.clustering import RootCauseClusterer
from src.models.phase1_rigor_evaluation import tier_features
from src.models.tier_e import TierEBlend


def _training_frame(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    """Builds whose failures follow the stream's last outcome, so history must matter."""
    rng = np.random.default_rng(seed)
    numeric, _ = tier_features("Enojobs")
    d = pd.DataFrame(rng.random((n, len(numeric))), columns=numeric)
    d["hist_stream_prev_failed"] = rng.integers(0, 2, n).astype(float)
    d.loc[rng.random(n) < 0.2, "hist_stream_prev_failed"] = np.nan
    d["target_failed"] = (rng.random(n) < np.where(d.hist_stream_prev_failed.eq(1), 0.7, 0.1)).astype(int)
    d["gh_lang"] = rng.choice(["ruby", "java"], n)
    d["gh_project_name"] = rng.choice(["acme/api", "acme/web"], n)
    d["gh_build_started_at"] = pd.Timestamp("2015-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    return d


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    path = tmp_path_factory.mktemp("tier_e_artifacts")
    numeric, categorical = tier_features("Enojobs")
    model = TierEBlend(numeric, categorical).fit(_training_frame())
    model.metadata = {"variant": "Enojobs_lgbmblend", "decision_threshold": 0.5}
    model.input_fill = {"gh_sloc": 0.25, "git_diff_src_churn": 0.75}
    joblib.dump(model, path / "tier_e_blend.joblib")
    matrix = pd.DataFrame([[0.0] * len(FEATURE_COLS), [1.0] * len(FEATURE_COLS)], columns=FEATURE_COLS)
    imputer = SimpleImputer(strategy="median").fit(matrix)
    joblib.dump({"imputer": imputer, "scaler": StandardScaler().fit(matrix), "feature_cols": FEATURE_COLS.copy(),
                 "cold_start_prior": 0.2, "decision_threshold": 0.9, "training_mode": "full"}, path / "preprocessing.joblib")
    joblib.dump({"prev_status_enc": LabelEncoder().fit(["__missing__"]), "lang_enc": LabelEncoder().fit(["__missing__"])},
                path / "encoders.joblib")
    AnomalyDetector(contamination=0.5).fit(imputer.transform(matrix)).save(str(path / "anomaly_detector.joblib"))
    RootCauseClusterer(n_clusters=2).fit(pd.Series(["database timeout", "database refused"])).save(
        str(path / "root_cause_clusterer.joblib"))
    (path / "best_model.txt").write_text("tier_e_blend\n", encoding="utf-8")
    return path


def test_predictor_scores_with_stream_history_and_own_threshold(artifacts) -> None:
    predictor = CICDPredictor(str(artifacts))
    assert predictor.is_tier_e and predictor.uses_stream_context
    assert predictor.decision_threshold == 0.5  # the blend's, not the forest's 0.9
    assert predictor.calibrator is None and predictor.preprocessing["training_mode"] == "tier_e:Enojobs_lgbmblend"
    base = {"git_diff_src_churn": 0.5, "gh_lang": "cobol", "git_branch": "master", "gh_is_pr": False}
    cold = predictor.predict_failure(base)["failure_probability"]
    red = predictor.predict_failure({**base, "hist_stream_prev_failed": 1})["failure_probability"]
    green = predictor.predict_failure({**base, "hist_stream_prev_failed": 0})["failure_probability"]
    assert 0 < green < red < 1 and 0 < cold < 1
    frame = predictor._tier_e_frame(base)
    assert frame.loc[0, "is_master"] == 1 and np.isnan(frame.loc[0, "hist_stream_completed"])
    # Omitted change inputs get the training median; supplied ones are kept.
    assert frame.loc[0, "gh_sloc"] == 0.25 and frame.loc[0, "git_diff_src_churn"] == 0.5
    assert np.isnan(frame.loc[0, "gh_team_size"])  # no median stored for it in this fixture
    result = predictor.full_analysis({**base, "hist_stream_prev_failed": 1}, include_explanation=True)
    assert result["explanation"] and all(np.isfinite(item["value"]) for item in result["explanation"])
    assert "hist_stream_prev_failed" in {item["feature"] for item in result["explanation"]}
    assert isinstance(result["anomaly"]["is_anomaly"], bool)


def test_blend_matches_saved_model_after_reload(artifacts) -> None:
    model = joblib.load(artifacts / "tier_e_blend.joblib")
    rows = _training_frame(50, seed=11)
    again = joblib.load(artifacts / "tier_e_blend.joblib")
    assert np.array_equal(model.predict_proba(rows), again.predict_proba(rows))
    contributions = model.contributions(rows)
    assert list(contributions.columns) == model.feature_cols and contributions.notna().all().all()


@pytest.fixture()
def client(artifacts):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with session_factory() as session:
        session.add(User(username="dev", hashed_password=get_password_hash("pw"), role=UserRole.DEVELOPER))
        session.commit()

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    predictor = CICDPredictor(str(artifacts))
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_predictor] = lambda: predictor
    with TestClient(app) as test_client:
        token = test_client.post("/auth/login", data={"username": "dev", "password": "pw"}).json()["access_token"]
        test_client.headers["Authorization"] = f"Bearer {token}"
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)


def test_api_derives_stream_history_and_in_flight_builds(client) -> None:
    branch = {"git_branch": "dev", "gh_is_pr": False, "gh_lang": "ruby"}
    first = client.post("/pipelines/acme/builds", json={"build_features": branch, "build_status": "failed"}).json()
    assert first["stream_context"]["hist_stream_completed"] == 0
    assert first["stream_context"]["hist_stream_prev_failed"] is None
    # A build scored without a reported status stores only the prediction: never history.
    client.post("/pipelines/acme/builds", json={"build_features": branch})
    running = client.post("/pipelines/acme/shadow-builds", json={"build_features": branch}).json()
    pr = client.post("/pipelines/acme/shadow-builds",
                     json={"build_features": {**branch, "gh_is_pr": True, "gh_pull_req_num": 7}}).json()

    context = client.post("/pipelines/acme/builds", json={"build_features": branch, "build_status": "passed"}).json()["stream_context"]
    assert context["hist_stream_completed"] == 1 and context["hist_stream_prev_failed"] == 1
    assert context["hist_repo_fail_rate_all"] == 1.0
    assert context["ctx_repo_pending"] == 2  # the branch shadow build and the PR shadow build
    assert context["ctx_stream_pending"] == 1 and context["ctx_prevbuild_pending"] == 1
    assert pr["stream_context"]["hist_stream_completed"] == 0  # a PR is its own stream

    outcome = client.patch(f"/pipelines/acme/shadow-builds/{running['build_log_id']}/outcome",
                           json={"build_status": "failed"})
    assert outcome.status_code == 200
    later = client.post("/pipelines/acme/shadow-builds", json={"build_features": branch}).json()["stream_context"]
    assert later["hist_stream_completed"] == 3 and later["hist_stream_fail_rate_all"] == pytest.approx(2 / 3)
    assert later["ctx_repo_pending"] == 1  # only the PR build is still running
