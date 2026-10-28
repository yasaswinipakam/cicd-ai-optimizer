"""Join pre-trigger commit metadata to finance suites and audit feature safety.

The generated taxonomy flags are descriptive-only in protocol v1.1. They are
saved for a transparent case-study overlay but deliberately excluded from any
predictor until a separate predeclared ablation is approved.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "data" / "processed" / "phase2_finance_v1_commit_suites.parquet"
COMMITS = ROOT / "data" / "raw" / "phase2_finance_v1_commits" / "commit_metadata.parquet"
OUT = ROOT / "data" / "processed" / "phase2_finance_v1_annotated_suites.parquet"
REPORTS = ROOT / "reports"

TAXONOMY = {
    "badge_payment_billing": ("payment", "billing", "invoice", "charge", "subscription", "checkout", "transfer"),
    "badge_ledger_accounting": ("ledger", "accounting", "journal", "transaction", "balance", "reconciliation"),
    "badge_auth_security": ("auth", "oauth", "login", "identity", "security", "credential", "token"),
    "badge_dependency_ci": ("package-lock", "yarn.lock", "pom.xml", "build.gradle", "gradle", "dockerfile", ".github/workflows"),
}


def main() -> None:
    suites = pd.read_parquet(SUITES)
    commits = pd.read_parquet(COMMITS)
    if commits.duplicated(["repository", "head_sha"]).any():
        raise AssertionError("commit metadata has duplicate suite keys")
    # Commit metadata is intentionally collected only for the predeclared
    # eligible repository set. Excluded cohort candidates remain in the raw
    # audit, not in this single-repository case-study table.
    suites = suites.loc[suites.repository.isin(commits.repository.unique())].copy()
    data = suites.merge(commits, on=["repository", "head_sha"], how="left", validate="one_to_one")
    fields = ["additions", "deletions", "total_changes", "changed_file_count", "changed_paths"]
    if data[fields].isna().any().any():
        raise AssertionError("missing commit metadata after merge")
    for badge, tokens in TAXONOMY.items():
        pattern = "|".join(re.escape(token) for token in tokens)
        data[badge] = data.changed_paths.str.lower().str.contains(pattern, regex=True, na=False)
    data["badge_any_finance_criticality"] = data[["badge_payment_billing", "badge_ledger_accounting", "badge_auth_security"]].any(axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT, index=False)

    feature_rows = [
        {"field": "additions", "source": "GitHub commit stats", "known_before_trigger": True, "predictor_v1_1": True, "risk": "commit exists before CI trigger"},
        {"field": "deletions", "source": "GitHub commit stats", "known_before_trigger": True, "predictor_v1_1": True, "risk": "commit exists before CI trigger"},
        {"field": "total_changes", "source": "GitHub commit stats", "known_before_trigger": True, "predictor_v1_1": True, "risk": "commit exists before CI trigger"},
        {"field": "changed_file_count", "source": "GitHub commit metadata", "known_before_trigger": True, "predictor_v1_1": True, "risk": "commit exists before CI trigger"},
        {"field": "path criticality badges", "source": "GitHub commit paths + frozen taxonomy", "known_before_trigger": True, "predictor_v1_1": False, "risk": "descriptive-only; no predictive claim"},
        {"field": "workflow names/count", "source": "post-trigger Actions metadata", "known_before_trigger": False, "predictor_v1_1": False, "risk": "excluded: workflow fan-out/outcomes can accrue after trigger"},
        {"field": "finish/duration", "source": "Actions updated_at", "known_before_trigger": False, "predictor_v1_1": False, "risk": "history boundary only; never a predictor"},
    ]
    availability = pd.DataFrame(feature_rows)
    availability.to_csv(REPORTS / "phase2_finance_feature_availability.csv", index=False)
    badge_counts = pd.DataFrame([{"badge": key, "commit_suites": int(data[key].sum()), "share": float(data[key].mean())} for key in [*TAXONOMY, "badge_any_finance_criticality"]])
    badge_counts.to_csv(REPORTS / "phase2_finance_path_badge_counts.csv", index=False)
    lines = [
        "# Phase 2 finance pilot: feature and taxonomy audit",
        "",
        f"Annotated commit/check-suites: {len(data):,}. Every suite has one matched public commit-metadata record and no duplicate suite key.",
        "",
        "## Feature availability",
        "",
        availability.to_markdown(index=False),
        "",
        "## Descriptive path badges",
        "",
        badge_counts.to_markdown(index=False, floatfmt=".3f"),
        "",
        "The path badges are not model features in v1.1. They are a descriptive overlay only, so any apparent relation to failure is not used to choose or improve the predictor.",
        "",
        "## Next permitted evaluation inputs",
        "",
        "The strict predictor may use log-transformed commit additions, deletions, total changes, changed-file count, calendar fields from trigger time, and finish-time-safe local history. It must exclude workflow fan-out fields, completion timestamps/duration, outcome-derived data, and the descriptive badges.",
    ]
    (REPORTS / "phase2_finance_feature_audit.md").write_text("\n".join(lines) + "\n")
    print(badge_counts.to_string(index=False))
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
