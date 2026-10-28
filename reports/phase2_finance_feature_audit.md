# Phase 2 finance pilot: feature and taxonomy audit

Annotated commit/check-suites: 1,787. Every suite has one matched public commit-metadata record and no duplicate suite key.

## Feature availability

| field                   | source                                | known_before_trigger   | predictor_v1_1   | risk                                                         |
|:------------------------|:--------------------------------------|:-----------------------|:-----------------|:-------------------------------------------------------------|
| additions               | GitHub commit stats                   | True                   | True             | commit exists before CI trigger                              |
| deletions               | GitHub commit stats                   | True                   | True             | commit exists before CI trigger                              |
| total_changes           | GitHub commit stats                   | True                   | True             | commit exists before CI trigger                              |
| changed_file_count      | GitHub commit metadata                | True                   | True             | commit exists before CI trigger                              |
| path criticality badges | GitHub commit paths + frozen taxonomy | True                   | False            | descriptive-only; no predictive claim                        |
| workflow names/count    | post-trigger Actions metadata         | False                  | False            | excluded: workflow fan-out/outcomes can accrue after trigger |
| finish/duration         | Actions updated_at                    | False                  | False            | history boundary only; never a predictor                     |

## Descriptive path badges

| badge                         |   commit_suites |   share |
|:------------------------------|----------------:|--------:|
| badge_payment_billing         |             489 |   0.274 |
| badge_ledger_accounting       |             426 |   0.238 |
| badge_auth_security           |             253 |   0.142 |
| badge_dependency_ci           |             599 |   0.335 |
| badge_any_finance_criticality |             795 |   0.445 |

The path badges are not model features in v1.1. They are a descriptive overlay only, so any apparent relation to failure is not used to choose or improve the predictor.

## Next permitted evaluation inputs

The strict predictor may use log-transformed commit additions, deletions, total changes, changed-file count, calendar fields from trigger time, and finish-time-safe local history. It must exclude workflow fan-out fields, completion timestamps/duration, outcome-derived data, and the descriptive badges.
