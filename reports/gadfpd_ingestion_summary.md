# GADFPD GitHub Actions Research Ingestion

This dataset is separate from TravisTorrent and from the deployed model. It contains only rows with a GitHub Actions `workflow_id` and an unambiguous success/failure conclusion.

- Labelled GitHub Actions runs: 146,110 across 32 repositories
- Failures: 8,942 (6.12%)
- Chronological range: 2025-02-24 12:58:46+00:00 to 2026-04-18 17:48:19+00:00
- Global chronological split: 116,888 train / 29,222 final test
- Repository holdout eligible: 26

`conclusion` is retained only as an audit label. `status`, `duration`, `run_started_at`, and `updated_at` are excluded from prediction research because they are available during or after execution.
