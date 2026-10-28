"""Merge completed Travis per-project online checkpoints without refitting.

The protocol is deliberately reported at the project level: every completed
repository contributes one mean across its five expanding-window test folds,
then projects—not rows or folds—are macro-averaged.  This prevents a very
large project from silently dominating the literature-comparison result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "reports" / "phase1_v2_parts"
REPORTS = ROOT / "reports"
SEED = 20261013
FOLDS = {6, 7, 8, 9, 10}


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def main() -> None:
    rows: list[dict] = []
    for path in sorted(PARTS.glob("travis_project_*_fold_*_C_rf_ros__finish_buffer_hours_*.json")):
        payload = json.loads(path.read_text())
        if not payload:
            continue
        row = dict(payload[0])
        if row.get("dataset") != "travis" or not str(row.get("split", "")).startswith("per_project_online_fold_"):
            continue
        row["source_file"] = path.name
        row["fold"] = int(str(row["split"]).rsplit("_", 1)[1])
        rows.append(row)
    folds = pd.DataFrame(rows)
    fold_out = REPORTS / "phase1_v2_travis_per_project_online_folds.csv"
    projects_out = REPORTS / "phase1_v2_travis_per_project_online_projects.csv"
    summary_out = REPORTS / "phase1_v2_travis_per_project_online_summary.md"
    if folds.empty:
        raise SystemExit("No completed C/RF/ROS per-project online checkpoints found.")
    folds.sort_values(["repository", "fold", "finish_buffer_hours"]).to_csv(fold_out, index=False)

    metric_columns = [
        "roc_auc", "pr_auc", "precision", "recall", "f1", "prevalence", "brier", "ece",
        "precision_at_10", "recall_at_10", "lift_at_10", "precision_at_15", "recall_at_15", "lift_at_15",
    ]
    grouped = folds.groupby(["repository", "finish_buffer_hours"], sort=True)
    projects = grouped[metric_columns].mean().reset_index()
    projects["completed_folds"] = grouped["fold"].nunique().to_numpy()
    projects["eligible_for_macro"] = projects["completed_folds"].eq(len(FOLDS))
    projects["total_test_builds"] = grouped["n"].sum().to_numpy()
    projects["total_test_failures"] = grouped["failures"].sum().to_numpy()
    projects.to_csv(projects_out, index=False)

    complete = projects.loc[projects.eligible_for_macro].copy()
    metrics = ["roc_auc", "pr_auc", "f1", "recall_at_10", "precision_at_10", "lift_at_10", "recall_at_15", "precision_at_15", "lift_at_15", "brier", "ece"]
    rng = np.random.default_rng(SEED)
    aggregate = []
    for metric in metrics:
        vals = complete[metric].to_numpy(float)
        low, high = bootstrap_mean(vals, rng)
        aggregate.append({"metric": metric, "macro_mean": np.nanmean(vals), "repo_bootstrap_ci_low": low, "repo_bootstrap_ci_high": high, "complete_projects": len(complete)})
    aggregate_df = pd.DataFrame(aggregate)
    aggregate_out = REPORTS / "phase1_v2_travis_per_project_online_macro.csv"
    aggregate_df.to_csv(aggregate_out, index=False)

    lines = [
        "# Travis per-project expanding online validation",
        "",
        "Each project is ordered chronologically, split into ten contiguous folds, and evaluated on folds 6–10 after fitting on folds 1..j-1. RandomOverSampler is applied to the training fold only. The macro result averages completed projects equally; its 95% interval bootstraps projects, not builds.",
        "",
        f"Completed checkpoints: {len(folds)} folds from {projects.repository.nunique()} projects. Complete five-fold projects included in the macro: {len(complete)}.",
        "",
        "Failure-class F1 must be read with prevalence; both are retained in the fold/project CSVs.",
        "",
        aggregate_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Projects with fewer than five completed folds remain in the project CSV but are excluded from the macro until their remaining units are run or explicitly skipped.",
    ]
    summary_out.write_text("\n".join(lines) + "\n")
    print(aggregate_df.to_string(index=False))
    print(f"Wrote {fold_out.relative_to(ROOT)}, {projects_out.relative_to(ROOT)}, {aggregate_out.relative_to(ROOT)}, {summary_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
