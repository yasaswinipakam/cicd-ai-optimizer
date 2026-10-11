"""Static safety checks for the privileged workflow_run log collector."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci-diagnostics-log-collector.yml"
POLLER = ROOT / ".github" / "workflows" / "ci-diagnostics-telemetry-poller.yml"


def test_log_collector_has_least_privilege_shape_and_never_executes_pr_code() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_run:" in workflow
    assert "actions: read" in workflow
    assert "actions/checkout" not in workflow
    assert "download-artifact" not in workflow
    assert "contents: write" not in workflow
    assert "X-CI-Diagnostics-Secret" in workflow
    assert "https://" in workflow or "DIAGNOSTICS_URL" in workflow


def test_active_telemetry_poller_is_scheduled_read_only_and_does_not_execute_pr_code() -> None:
    workflow = POLLER.read_text(encoding="utf-8")
    assert "schedule:" in workflow and "workflow_dispatch:" in workflow
    assert "actions: read" in workflow
    assert "actions/checkout" not in workflow
    assert "download-artifact" not in workflow
    assert "contents: write" not in workflow
    assert "/webhooks/github-actions/job-telemetry" in workflow
