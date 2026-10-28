# External test of the frozen Tier E recipe on GitHub Actions — protocol

Written on 2026-10-28, before any model was fit on this data.

The evaluation script records this file's SHA-256 in every output, so later
edits to the protocol are detectable.

## Question

Does the Tier E recipe beat Tier C RF on fresh data from a different CI
platform? The recipe and its comparison were frozen on Travis folds 6–10.

- **Recipe:** the LightGBM binary + `rank_xendcg` blend, fed with
  finish-time-safe stream history and in-flight context.
- **Travis result (folds 7–10):** within-repo AUC 0.695 (no job counts) vs
  0.628 for Tier C RF.

Nothing below may be changed after results are seen. Any later deviation is
reported as a dated amendment.

## What was inspected beforehand

Only descriptive statistics:

- row counts by conclusion, event and attempt;
- failure rate by event and by repository;
- wall-clock time from creation to last update, by conclusion;
- number of runs per repository and per commit.

No model, score or metric was computed on this data for the Tier E question.

An earlier baseline existed: an all-events, 80/20 GHA random forest with
pooled AUC 0.80. It used a different population and features, so it is not
the comparator here.

## Data and unit

- **Source:** `data/raw/final_research_dataset_MASTER.csv`.
- **Unit:** one workflow run with `run_attempt == 1`.
  - Re-run runs are therefore absent, as in the existing GHA baseline. This
    is a known selection limitation.
- **Events:** `push` and `pull_request` only, the code-change triggers that
  correspond to Travis builds. Other events (`issue_comment`, `status`,
  `schedule`, …) almost never fail and would let a model score well just by
  telling event types apart.
- **Label:** 1 if the conclusion is `failure` or `startup_failure`; 0 if
  `success`. All other conclusions are dropped.
- **Completion time:** `updated_at`.
  - Runs whose `updated_at` is more than 35 days after `created_at` are
    dropped (687 runs, 1%). GitHub's maximum run time is 35 days, so these
    timestamps were touched later and the true completion time is unknown.
  - The rule applies to both classes.
- **Trigger time:** `created_at`. Ties are broken by `run_id`.

## Mapping Tier E onto GitHub Actions

| Travis Tier E | GitHub Actions analogue |
|---|---|
| repository | `repo` |
| stream: PR number, or branch for pushes | PR runs: `(actor_login, head_branch)`; push runs: `head_branch`. GitHub Actions data has no PR number. The actor separates fork PRs that share a branch name such as `main`. |
| `tr_prev_build` (previous build on the same branch) | previous run of the same `workflow_id` on the same stream |
| finish = start + duration | finish = `updated_at` (plus buffer) |
| job count | not available, so the no-job-count variant (`Enojobs`, the deployed one) is used |
| Tier A change features | `additions`, `deletions`, `total_churn`, `files_modified`, `msg_len`, `is_merge`, `num_parents`, `time_since_last_commit`, hour, weekday, month, weekend, night (UTC), `is_pr` (event = pull_request), `is_default_branch` (`head_branch` in main/master) |
| `gh_lang` (categorical) | `workflow_id` (categorical) |
| Tier B repository fields | none available |

History and in-flight columns are computed by the unmodified Travis functions
`add_safe_history` and `add_stream_history`, run on the renamed columns. They
are re-verified by the same brute-force check, `_assert_stream_history`.

## Feature sets and models

All models use the identical base features (change + time + workflow):

- **A_rf:** base features only.
- **C_rf:** base + repository history (`HISTORY`). This is the Tier C
  comparator.
- **E_lgbmblend:** base + `HISTORY` + `STREAM` + `INFLIGHT`.
  - Same LightGBM parameters, repo×month rank queries, z-blend and Platt map
    as on Travis.
- **Baselines:** `C_historyonly` (last-5 repository failure rate) and
  `repo_prior`.
- **Attribution, reported but never used for selection:** `C_lgbmblend`
  (the ranking recipe without stream/in-flight features) and `E_lgbm`
  (binary objective only).

## Split

Walk-forward, matching Travis:

- the data, sorted by trigger time, is cut into 10 chronological folds;
- test fold k is scored by a model trained on folds 1…k-1, for k = 6…10;
- there is no development fold, because nothing is tuned here. Folds 6–10
  are all confirmation.

## Primary result

The mean over folds 6–10 of macro within-repo AUC, E_lgbmblend minus C_rf, at
a 0 h finish buffer.

- **Uncertainty:** each fold's paired repository-cluster bootstrap 95% CI
  (1,000 resamples, the same code as Travis).
- **Reading:** the recipe transfers if the mean delta is > 0 and at least 4
  of the 5 fold CIs exclude 0.
  - A mean delta > 0 with fewer CIs above 0 is reported as "inconclusive".
  - A delta ≤ 0 is reported as "does not transfer".

## Secondary results (all reported, none selected)

- The same comparison for recall@15% and lift@15% within repositories, and
  for pooled AUC.
- A 6 h finish buffer.
- **New breakages:** runs whose previous same-stream, same-workflow run passed,
  as in the Travis new-breakage rescoring.
- **Label-validity sensitivity:** drop `startup_failure` runs and failures
  with zero recorded duration. These are proxies for runs that never
  executed, which is the artifact found in the Fineract pilot.
- A_rf, the history-only baseline and the repository prior, for context.

## Known limitations, stated in advance

- Only 30 repositories (about 25 with both classes per test fold), so CIs
  will be wide.
- The data covers 2025–2026 only.
- First-attempt filtering excludes runs that were re-run.
- The PR stream key is an approximation.
- One CI run here is a single workflow, whereas a Travis build covered a
  whole build matrix.
