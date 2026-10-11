"""Read-only live shadow-mode monitoring for completed CI build outcomes.

Run this only after shadow records have accumulated outcomes. It never alters a
prediction, decision threshold, CI build, or database row; it produces evidence
for deciding whether developer-facing alerts are useful.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import joinedload

from api.db.database import SessionLocal
from api.db.models import BuildLog, Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def _rows_from_database() -> pd.DataFrame:
    session = SessionLocal()
    try:
        logs = (
            session.query(BuildLog)
            .options(joinedload(BuildLog.pipeline), joinedload(BuildLog.predictions))
            .all()
        )
        rows = []
        for log in logs:
            context = (log.raw_features or {}).get("_prediction_context", {})
            if not context.get("shadow_mode") or log.build_status not in {"passed", "failed"}:
                continue
            prediction = log.predictions[0] if log.predictions else None
            if prediction is None:
                continue
            raw = log.raw_features or {}
            rows.append(
                {
                    "timestamp": pd.Timestamp(log.timestamp),
                    "pipeline": log.pipeline.name if log.pipeline else None,
                    "language": raw.get("gh_lang", "unknown"),
                    "actual_failed": int(log.build_status == "failed"),
                    "probability": float(prediction.failure_probability),
                    "threshold": float(context.get("decision_threshold", 0.5)),
                    "alert": int(prediction.failure_probability >= float(context.get("decision_threshold", 0.5))),
                    "model_version": context.get("model_artifact_sha256"),
                    "calibration_method": context.get("calibration_method", "raw"),
                    "calibration_version": context.get("calibration_artifact_sha256"),
                }
            )
        return pd.DataFrame(rows)
    finally:
        session.close()


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"resolved_builds": 0, "actual_failure_rate": 0.0, "alert_rate": 0.0, "tier_precision": 0.0, "failure_capture_rate": 0.0}
    actual = frame["actual_failed"]
    alerts = frame["alert"]
    return {
        "resolved_builds": int(len(frame)),
        "actual_failure_rate": float(actual.mean()),
        "alert_rate": float(alerts.mean()),
        "tier_precision": float(actual.loc[alerts.eq(1)].mean()) if alerts.any() else 0.0,
        "failure_capture_rate": float(alerts.loc[actual.eq(1)].mean()) if actual.any() else 0.0,
    }


def _grouped_metrics(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[group, "resolved_builds", "actual_failure_rate", "alert_rate", "tier_precision", "failure_capture_rate"])
    return pd.DataFrame(
        [{group: name, **_metrics(part)} for name, part in frame.groupby(group, dropna=False)]
    ).sort_values("resolved_builds", ascending=False)


def write_shadow_monitoring_report(output_dir: Path = REPORTS_DIR) -> dict[str, Path]:
    """Summarize completed shadow predictions by time, repo, and language."""
    frame = _rows_from_database()
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = _metrics(frame)
    if not frame.empty:
        frame["month"] = frame["timestamp"].dt.to_period("M").astype(str)
    by_month = _grouped_metrics(frame, "month")
    by_repository = _grouped_metrics(frame, "pipeline")
    by_language = _grouped_metrics(frame, "language")
    paths = {
        "overall": output_dir / "shadow_monitoring_summary.csv",
        "monthly": output_dir / "shadow_monitoring_by_month.csv",
        "repository": output_dir / "shadow_monitoring_by_repository.csv",
        "language": output_dir / "shadow_monitoring_by_language.csv",
        "analysis": output_dir / "shadow_monitoring.md",
    }
    pd.DataFrame([overall]).to_csv(paths["overall"], index=False)
    by_month.to_csv(paths["monthly"], index=False)
    by_repository.to_csv(paths["repository"], index=False)
    by_language.to_csv(paths["language"], index=False)
    if frame.empty:
        text = "# Shadow Monitoring\n\nNo completed shadow records are available yet. Continue non-blocking collection; pending builds are intentionally excluded.\n"
    else:
        text = (
            "# Shadow Monitoring\n\n"
            f"Completed shadow builds: {overall['resolved_builds']:,}. Actual failure rate: {overall['actual_failure_rate']:.2%}. "
            f"Observed alert rate: {overall['alert_rate']:.2%}. Tier precision: {overall['tier_precision']:.2%}. "
            f"Failure capture: {overall['failure_capture_rate']:.2%}.\n\n"
            "The CSV breakdowns report these values by month, repository, and language. Interpret small groups cautiously; this is operational monitoring, not a basis for automatic CI blocking.\n\n"
            "Feature-distribution drift needs a stable, versioned live-data baseline and enough outcomes. This report records model/calibrator versions for that later comparison but does not infer drift from sparse shadow data.\n"
        )
    paths["analysis"].write_text(text, encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a read-only shadow-mode monitoring report.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    for label, path in write_shadow_monitoring_report(args.output_dir).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
