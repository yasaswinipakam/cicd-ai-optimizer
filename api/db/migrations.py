"""Small schema migration runner for the prototype's SQLAlchemy schema.

Migrations 001-003 add tables. Migration 004 removes sign-in: it drops the
pipeline owner column and the accounts table while keeping every pipeline,
build, prediction and shadow record.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, MetaData, String, Table, inspect, select, text

from api.db.database import engine
from api.db.models import JobFailureAnalysis, JobRun, JobStepRun, Pipeline, ProviderWorkflowOutcome, WorkflowRun, utc_now


MIGRATIONS = {
    "001_workflow_job_telemetry": (WorkflowRun.__table__, JobRun.__table__, JobStepRun.__table__),
    "002_job_failure_analysis": (JobFailureAnalysis.__table__,),
    "003_provider_workflow_outcomes": (ProviderWorkflowOutcome.__table__,),
    "004_remove_accounts": (),
}


def _remove_accounts(connection) -> None:
    """Drop pipelines.owner_id and the users table from databases created with sign-in."""
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    if "pipelines" in tables and "owner_id" in {c["name"] for c in inspector.get_columns("pipelines")}:
        if connection.dialect.name == "sqlite":
            # SQLite cannot drop a foreign-key column, so rebuild the table with
            # the same ids; other tables' references to "pipelines" stay valid.
            keep = ", ".join(column.name for column in Pipeline.__table__.columns)
            Pipeline.__table__.to_metadata(MetaData(), name="pipelines_new").create(bind=connection)
            connection.execute(text(f"INSERT INTO pipelines_new ({keep}) SELECT {keep} FROM pipelines"))
            connection.execute(text("DROP TABLE pipelines"))
            connection.execute(text("ALTER TABLE pipelines_new RENAME TO pipelines"))
        else:
            connection.execute(text("ALTER TABLE pipelines DROP COLUMN owner_id"))
    if "users" in tables:
        connection.execute(text("DROP TABLE users"))


def apply_migrations() -> list[str]:
    """Create missing additive tables and record each applied schema version."""
    metadata = MetaData()
    versions = Table(
        "schema_migrations", metadata,
        Column("version", String(100), primary_key=True),
        Column("applied_at", DateTime(timezone=True), nullable=False, default=utc_now),
    )
    metadata.create_all(bind=engine)
    applied: list[str] = []
    with engine.begin() as connection:
        existing = set(connection.execute(select(versions.c.version)).scalars())
        for version, tables in MIGRATIONS.items():
            for table in tables:
                table.create(bind=connection, checkfirst=True)
            if version == "004_remove_accounts":
                _remove_accounts(connection)
            if version not in existing:
                connection.execute(versions.insert().values(version=version, applied_at=utc_now()))
                applied.append(version)
    return applied
