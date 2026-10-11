"""Relationship tests for the SQLAlchemy persistence models."""

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.db.database import Base
from api.db.models import BuildLog, Pipeline, Prediction, User, UserRole


def test_build_audit_relationships_resolve_with_in_memory_sqlite() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        user = User(username="developer", hashed_password="bcrypt-hash", role=UserRole.DEVELOPER)
        pipeline = Pipeline(name="main-ci", status="active", owner=user)
        build_log = BuildLog(
            pipeline=pipeline,
            timestamp=datetime.now(timezone.utc),
            log_content="unit test failure",
            build_status="failed",
            stage_durations={"test": 32.1},
            raw_features={"gh_lang": "python"},
        )
        prediction = Prediction(
            build_log=build_log,
            model_name="xgboost",
            failure_probability=0.82,
            predicted_status="failed",
            is_anomaly=True,
            anomaly_score=-0.14,
            root_cause_cluster_id=2,
            root_cause_label="database / timeout",
        )
        session.add(prediction)
        session.commit()
        session.expire_all()

        persisted_prediction = session.query(Prediction).one()
        assert persisted_prediction.build_log.pipeline.owner.username == "developer"
        assert persisted_prediction.build_log.pipeline.name == "main-ci"
        assert persisted_prediction.build_log.predictions[0].id == persisted_prediction.id
    finally:
        session.close()
        Base.metadata.drop_all(engine)
