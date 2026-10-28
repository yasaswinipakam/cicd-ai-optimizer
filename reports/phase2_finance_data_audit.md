# Phase 2 finance pilot: pre-modelling data audit

## Collection

Retrieved: 2026-10-14T23:25:55.648825+00:00. Raw workflow-run records: 81,027. Repositories with an explicit GitHub pagination ceiling: actualbudget/actual, apache/fineract.

The raw pages and their SHA-256 hashes are retained in `data/raw/phase2_finance_v1/`. A pagination ceiling means the extract is the newest accessible window, not a claim of complete repository history.

## Raw integrity and provisional first-attempt filter

| repository              |   raw_workflow_runs |   unique_run_ids | earliest_raw_trigger      | latest_raw_trigger        |   duplicate_run_ids |   provisional_first_attempt_runs |   primary_failures |   startup_failures |   unique_head_shas | earliest_retained_trigger   | latest_retained_trigger   |   retained_span_days |   additional_runs_sharing_sha |   sha_reuse_share | provisional_workflow_run_eligible   |   provisional_final_20_runs |   provisional_final_20_failures |
|:------------------------|--------------------:|-----------------:|:--------------------------|:--------------------------|--------------------:|---------------------------------:|-------------------:|-------------------:|-------------------:|:----------------------------|:--------------------------|---------------------:|------------------------------:|------------------:|:------------------------------------|----------------------------:|--------------------------------:|
| actualbudget/actual     |               40000 |            40000 | 2026-07-16 20:24:16+00:00 | 2026-10-14 23:23:26+00:00 |                   0 |                            19912 |                728 |                  0 |               2186 | 2026-07-16 20:25:56+00:00   | 2026-10-14 23:17:48+00:00 |                   90 |                         17726 |             0.890 | False                               |                        3983 |                             142 |
| apache/fineract         |               40000 |            40000 | 2026-02-02 16:55:26+00:00 | 2026-10-14 17:18:03+00:00 |                   0 |                            10253 |               2561 |                260 |               1787 | 2026-02-02 17:44:09+00:00   | 2026-10-14 16:42:37+00:00 |                  253 |                          8466 |             0.826 | True                                |                        2051 |                            1420 |
| firefly-iii/firefly-iii |                 482 |              482 | 2025-09-01 04:00:06+00:00 | 2026-10-14 18:46:10+00:00 |                   0 |                               66 |                 12 |                  0 |                 66 | 2026-05-25 04:22:14+00:00   | 2026-10-14 18:46:10+00:00 |                  142 |                             0 |             0.000 | False                               |                          14 |                              12 |
| killbill/killbill       |                 545 |              545 | 2025-09-17 07:25:59+00:00 | 2026-10-12 23:02:48+00:00 |                   0 |                              428 |                103 |                  5 |                210 | 2025-09-17 07:25:59+00:00   | 2026-10-12 23:02:48+00:00 |                  390 |                           218 |             0.509 | False                               |                          86 |                              13 |

The provisional filter is completed first-attempt `push`/`pull_request` workflow runs with conclusion success, failure, or startup_failure. `failure` is the primary label; `startup_failure` remains separate. Cancelled, skipped, action-required, queued, scheduled, manual, and non-CI event records are retained in raw data but excluded from the provisional analytic filter.

## Blocking unit-of-analysis decision

Workflow-run IDs are unique, but many retained workflow runs share a commit SHA. Therefore workflow runs cannot yet be described as independent code changes. Splitting or averaging them as independent build-like units could overweight matrix/workflow-rich commits and allow the same change to occur on both sides of a temporal boundary.

Before modelling, choose one pre-specified unit:

1. **Commit/check-suite unit (recommended):** group first-attempt eligible runs by repository and head SHA; label failed if any constituent workflow run failed; start at the earliest constituent trigger and finish at the latest constituent completion. This aligns the prediction unit with a code change and prevents same-SHA cross-split duplication.
2. **Workflow-run unit:** retain individual runs, but purge every training run sharing a test head SHA and prohibit same-SHA outcome history. This answers a narrower question about ranking CI checks, not code changes.

No model fitting, calibration, feature selection, or result table is authorized until this decision is recorded. The provisional workflow-run eligibility column is not a final eligibility result under the recommended commit/check-suite unit.

## Files produced

- `reports/phase2_finance_candidate_eligibility.csv`
- `reports/phase2_finance_raw_outcomes.csv`
- `reports/phase2_finance_raw_events.csv`
