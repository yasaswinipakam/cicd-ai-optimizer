# Phase 2 finance-software pilot: frozen protocol v1.1

## Status and boundary

This is a separate, small external-validity pilot. It does not alter the
TravisTorrent/GADFPD Phase 1 model selection or headline. The Phase 1
per-project result currently remains a large-repository convenience pilot;
it is not combined with this cohort.

The purpose is deliberately narrow:

> On a predeclared group of public finance-software repositories, does a
> chronological, build/run-level risk ranking capture more failed checks than
> random review at a fixed 15% alert budget?

A null result, a cohort in which no repository is eligible, or material
heterogeneity between repositories is a valid result.

## Predeclared candidate cohort

The complete v1 cohort is recorded in
`reports/phase2_finance_candidate_repositories.csv`, retrieved on 2026-10-15:

1. `apache/fineract` — core banking and microfinance.
2. `killbill/killbill` — subscription billing and payments.
3. `firefly-iii/firefly-iii` — personal finance management.
4. `actualbudget/actual` — personal finance and budgeting.

All are public repositories selected from their stated finance/billing domain
and visible GitHub Actions presence, not from failure rates or model results.
No repository may be substituted after outcome collection. A candidate that
does not meet the eligibility rule is retained in the exclusions table with
its reason.

## Collection boundary

Collect public GitHub Actions **workflow-run metadata**, not logs, for each
candidate at one recorded retrieval time. Preserve the unmodified response or
its content hash, pagination metadata, and collection command/config. Required
fields are repository, workflow/run identifiers, `run_attempt`, event, branch,
head SHA, creation/start/update timestamps, status, conclusion, and workflow
name. The dataset must state whether timestamps are API creation or actual
run-start times; if actual run-start is unavailable, call the field
`trigger_time` and do not imply a more precise boundary.

The GitHub endpoint may enforce a pagination ceiling. If it does, retain the
newest successfully retrieved records, preserve the terminal error response,
and record the repository as **historically truncated** in the collection
manifest and data audit. Do not call the extract complete or backfill older
records by changing the endpoint/query after observing outcomes. The analysis
window is then the retained newest observation window, whose exact start and
end dates must be reported per repository.

Keep only completed, first-attempt (`run_attempt == 1`) CI workflow runs whose
event is `push` or `pull_request`. Exclude scheduled, manual, release,
cancelled, skipped, neutral, and action-required runs. Preserve those excluded
records and counts in a collection audit. The primary label is
`conclusion == "failure"`. If an API export contains a distinct
`startup_failure` conclusion, report it separately and do not silently merge
it with ordinary test/build failures.

The prediction unit is a **commit/check-suite**: all retained first-attempt
workflow runs with the same repository and head SHA are collapsed into one
unit. Its label is failed if any constituent workflow has `failure`; its
trigger is the earliest constituent creation time; and its finish is the
latest constituent completion/update time. `startup_failure` remains a
separate flag rather than being silently merged into the primary label.

This v1.1 amendment was made after the pre-modelling data audit found extensive
same-SHA workflow fan-out. It prevents matrix/workflow-rich commits from being
counted as independent changes or crossing a temporal split as duplicate code
changes. No model metric was observed before this amendment.

## Eligibility (applied after collection, before model fitting)

A candidate is eligible only if its retained first-attempt data contain:

- at least 180 calendar days between the earliest and latest trigger time;
- at least 500 retained workflow runs;
- at least 30 primary-label failures; and
- at least 10 failures in the chronologically final 20% development-test
  window.

These thresholds are safeguards for reporting uncertainty, not tuning rules.
No threshold, feature, repository, or time window may be changed after reading
model metrics.

## Leakage-safe features and time boundary

Every feature must be available strictly before `trigger_time`.

Allowed initial feature family:

- calendar fields derived from trigger time;
- declared event/branch/workflow metadata when present at trigger time;
- source-controlled path criticality labels defined below; and
- local history calculated from prior **completed** runs whose finish time is
  strictly before the current run's trigger time.

If finish time is unavailable, use the recorded completion/update timestamp
only after validating that it is at or after run completion. Missing or invalid
durations are never treated as zero. Use the per-repository p95 duration, then
the global p95, as a conservative fallback; log fallback counts. Evaluate
finish buffers of 0, 1, and 6 hours as a sensitivity analysis.

History is restricted to previous completed runs: previous outcome; failure
rate over the last 5 completed runs; and failure rate across all earlier
completed runs. It must be recomputed independently inside every temporal
split. Repository-wide counts, eventual run outcome, duration, updated-at
values, issue comments, stars, forks, and any post-trigger information are
not eligible features.

## Criticality overlay (descriptive, not predictive in v1)

Path labels are derived from the commit/workflow source tree using this frozen
taxonomy. They are displayed in a case-study table or UI only and are not
added to the v1 predictor:

| Badge | Case-insensitive source-path tokens |
|---|---|
| payment/billing | `payment`, `billing`, `invoice`, `charge`, `subscription`, `checkout`, `transfer` |
| ledger/accounting | `ledger`, `accounting`, `journal`, `transaction`, `balance`, `reconciliation` |
| authentication/security | `auth`, `oauth`, `login`, `identity`, `security`, `credential`, `token` |
| dependency/CI configuration | `package-lock`, `yarn.lock`, `pom.xml`, `build.gradle`, `gradle`, `dockerfile`, `.github/workflows` |

The taxonomy is not ground truth for financial criticality, does not diagnose
failures, and is not used to select repositories or change outcomes.

## Evaluation and selection firewall

For each eligible repository:

1. Order retained run-level units by trigger time, with a deterministic run-ID
   tie breaker.
2. Keep the final 20% as a development test, never used for feature selection,
   resampling, calibration, or threshold selection.
3. Use an earlier validation window to select between explicitly named
   calibration options by minimum Brier score; record every candidate and the
   validation prevalence. Use a fixed 15% ranking budget for headline alert
   metrics, not a test-chosen probability threshold.
4. Separately perform ten contiguous expanding folds, testing folds 6–10 and
   training on folds 1..j-1. Resample only the training fold if imbalance
   handling is used.

The primary model is the existing fixed 200-tree Random Forest configuration.
The comparable baselines are majority, previous-completed-outcome, last-five
completed-run failure rate, repository-prior, and logistic regression.
No Phase 1 feature/model change may be selected from this cohort.

Report per repository: retained runs, failures, prevalence, ROC-AUC, PR-AUC,
failure-class precision/recall/F1 (with prevalence), Brier, ECE, and
precision/recall/lift at 10% and 15%. Use repository macro averages only for
descriptive cohort summaries. Use weekly-block bootstrap intervals for
repository-specific ranking metrics and repository-cluster bootstrap intervals
for any multi-repository average.

## Required artifacts before results

- `phase2_finance_collection_manifest.json`: retrieval date/time, API query,
  pagination, raw-response hashes, code version, and exclusions.
- `phase2_finance_data_audit.md`: candidate/eligibility counts, duplicate and
  retry policy checks, duration quality, label distribution, and time range.
- versioned run-parts, saved predictions, and a configuration manifest,
  following the Phase 1 checkpoint convention.
- `phase2_finance_results.csv` and a short markdown report that clearly labels
  this a four-candidate pilot.

## Amendment v1.2 (2026-10-28): label and timing correction, stream history

**Why.** A post-result data review found Fineract first-attempt runs recorded
as `failure` that executed zero jobs (1,209 failure runs showed 1–30 days
between start and `updated_at`; every sampled one had zero jobs; exact counts
are in the v1.2 results). They were pull-request runs
waiting for maintainer approval that GitHub later expired (up to the 30-day
limit) or failed when the PR closed. Their `updated_at` is that expiry, not a
build completion, so these "failures" were not code outcomes and became known
weeks after trigger. This biased both the label and the finish-time-safe
history (recent failures looked unfinished far longer than passes).

**Disclosure.** v1.1 and the post-hoc v2.0 results were seen before this
amendment. v1.2 corrects a measurement defect; it is not blind, and v1.1
remains reported alongside it.

**Rules fixed before any v1.2 metric is computed:**

1. A run concluded `failure` or `startup_failure` counts toward a
   commit/check-suite only if it executed at least one job (job lists from the
   GitHub jobs API, retrieved once and stored in
   `data/raw/phase2_finance_v1_jobs/`). Zero-job failed runs are dropped; a
   suite with no remaining runs is dropped. The check is limited to failed
   runs because the defect (expiry of unapproved runs) can only manufacture
   failures, never successes; successful runs are kept as in v1.1.
2. Suite label: failed if any remaining constituent run has conclusion
   `failure`. `startup_failure` stays a separate flag. Trigger is the earliest
   remaining `created_at`; finish is the latest remaining `updated_at`.
3. Eligibility thresholds, splits, calibration rule, RF configuration, finish
   buffers (0/1/6 h) and the 15% budget are unchanged from v1.1.
4. Primary model: the v1.1 RF on the v1.1 feature set (`C_rf`) with corrected
   labels and timing.
5. One pre-specified secondary variant (`Cstream_rf`): the same RF plus
   finish-time-safe **stream** history, where a stream is the run's
   `head_branch` (the PR source branch, or the pushed branch). Features:
   previous completed stream outcome, last-5 stream failure rate, stream
   completed count. No other variants; path badges remain descriptive only.
6. Results go to `reports/phase2_finance_v1_2_parts/` and are reported in full,
   including a null result.

## Interpretation limits

Do not describe this pilot as evidence about banks, production financial
institutions, security incidents, prevented failures, monetary savings, or all
finance software. It concerns public open-source repositories and their
public CI metadata only.
