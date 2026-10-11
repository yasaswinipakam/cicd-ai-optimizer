"""Small additive schema migration runner for the prototype's SQLAlchemy schema.

The project currently has no destructive schema changes. This registry records
additive migrations so an existing SQLite demo database receives telemetry tables
without deleting users, builds, predictions, or shadow records.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, MetaData, String, Table, select

from api.db.database import engine
from api.db.models import JobFailureAnalysis, JobRun, JobStepRun, ProviderWorkflowOutcome, WorkflowRun, utc_now


MIGRATIONS = {
    "001_workflow_job_telemetry": (WorkflowRun.__table__, JobRun.__table__, JobStepRun.__table__),
    "002_job_failure_analysis": (JobFailureAnalysis.__table__,),
    "003_provider_workflow_outcomes": (ProviderWorkflowOutcome.__table__,),
}


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
            if version not in existing:
                connection.execute(versions.insert().values(version=version, applied_at=utc_now()))
                applied.append(version)
    return applied
