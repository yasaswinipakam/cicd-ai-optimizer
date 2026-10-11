# GADFPD Schema and Availability Audit

- File: `final_research_dataset_MASTER.csv`
- Rows: 303,079; columns: 38; repositories: 654
- `created_at` range: 2016-08-08 17:58:18+00:00 to 2026-04-18 17:48:19+00:00
- Conclusion counts: {'success': 137168, 'skipped': 38041, 'failure': 8671, 'cancelled': 5260, 'startup_failure': 271, 'action_required': 35, '<missing>': 91047, 'passed': 17522, 'failed': 3702, 'errored': 1253, 'canceled': 109}
- Scope: separate GitHub Actions research data; no merge with TravisTorrent and no live-model change.

| Field | Sample dtype | Missing | Disposition | Reason |
| --- | --- | ---: | --- | --- |
| `run_id` | int64 | 0.00% | excluded_or_diagnostic | Opaque execution identifier with no stable predictive semantics. |
| `repo` | object | 0.00% | candidate_pre_run | Repository identity; use for group-aware evaluation and optional local calibration, not as a global-model feature. |
| `repo_name` | float64 | 92.55% | excluded_or_diagnostic | Duplicate/mostly missing repository field; use canonical repo only after validation. |
| `commit_sha` | object | 0.00% | excluded_or_diagnostic | Opaque commit identifier; use only for joins and deduplication. |
| `conclusion` | object | 30.04% | excluded_or_diagnostic | Post-run outcome; this is the research target, never a predictive feature. |
| `status` | object | 7.46% | excluded_or_diagnostic | Execution state becomes known during/after workflow execution; exclude from pre-run prediction. |
| `duration` | float64 | 0.00% | excluded_or_diagnostic | Observed only after or during execution; post-build leakage. |
| `created_at` | object | 30.04% | candidate_pre_run | Run creation time; derive calendar features only. |
| `run_started_at` | object | 30.04% | excluded_or_diagnostic | Execution-start timestamp; retain for chronology only, not a pre-dispatch feature. |
| `updated_at` | object | 37.49% | excluded_or_diagnostic | Updated after workflow progress/completion; post-build leakage. |
| `event` | object | 37.49% | candidate_pre_run | Workflow trigger is known when the run is created. |
| `head_branch` | object | 30.04% | candidate_pre_run | Branch is known at workflow dispatch. |
| `run_number` | float64 | 30.04% | candidate_pre_run | Workflow run sequence is assigned at run creation. |
| `author_name` | object | 37.49% | excluded_or_diagnostic | Personally identifying and brittle categorical field; do not use as a global feature. |
| `author_email` | object | 37.49% | excluded_or_diagnostic | Personally identifying field; do not use. |
| `committer_name` | object | 37.49% | excluded_or_diagnostic | Personally identifying and brittle categorical field; do not use. |
| `committer_email` | object | 37.49% | excluded_or_diagnostic | Personally identifying field; do not use. |
| `authored_date` | object | 37.49% | excluded_or_diagnostic | Raw commit timestamp; derive calendar features from a documented timestamp instead. |
| `committed_date` | object | 37.49% | excluded_or_diagnostic | Raw commit timestamp; derive calendar features from a documented timestamp instead. |
| `message` | object | 37.49% | excluded_or_diagnostic | Raw text can be studied separately, but is excluded from the first tabular baseline to avoid fragile lexical leakage. |
| `msg_len` | int64 | 0.00% | candidate_pre_run | Commit-message length is known before the workflow starts. |
| `is_merge` | bool | 37.49% | candidate_pre_run | Merge-commit status is known from commit metadata before execution. |
| `num_parents` | float64 | 37.49% | candidate_pre_run | Commit graph structure is known before execution. |
| `additions` | int64 | 0.00% | candidate_pre_run | Commit-level additions are available from the commit before the run starts. |
| `deletions` | int64 | 0.00% | candidate_pre_run | Commit-level deletions are available from the commit before the run starts. |
| `total_churn` | float64 | 37.49% | candidate_pre_run | Commit-level churn is available before the run starts. |
| `files_modified` | int64 | 0.00% | candidate_pre_run | Changed-file count is available before the run starts. |
| `timestamp_raw` | object | 0.00% | excluded_or_diagnostic | Duplicate raw timestamp field; use only to validate chronology. |
| `commit_date` | float64 | 69.96% | excluded_or_diagnostic | Duplicate/mostly missing timestamp field; exclude pending data-quality validation. |
| `author` | float64 | 62.51% | excluded_or_diagnostic | Duplicate/mostly missing author field; exclude for privacy and quality reasons. |
| `workflow_id` | float64 | 37.49% | candidate_pre_run | Workflow configuration identity is known at dispatch; assess generalization carefully. |
| `node_id` | object | 37.49% | excluded_or_diagnostic | Opaque platform identifier; never a model feature. |
| `head_sha` | object | 37.49% | excluded_or_diagnostic | Opaque commit identifier; use only for joins and deduplication. |
| `run_attempt` | float64 | 37.49% | excluded_or_diagnostic | May encode retries following an earlier failure; exclude absent a proven pre-run definition. |
| `actor_login` | object | 37.49% | excluded_or_diagnostic | Platform actor identifier; high-cardinality and repository-specific; exclude from the global baseline. |
| `triggering_actor_login` | object | 37.49% | excluded_or_diagnostic | Platform actor identifier; high-cardinality and repository-specific; exclude from the global baseline. |
| `time_since_last_commit` | float64 | 37.49% | candidate_pre_run | Historical commit interval can be computed before dispatch. |
| `timestamp` | float64 | 69.96% | excluded_or_diagnostic | Duplicate/mostly missing timestamp field; exclude pending data-quality validation. |
