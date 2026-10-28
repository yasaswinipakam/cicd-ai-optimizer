# Confirmation test of the frozen Tier E recipe on new GitHub Actions repositories — protocol

Written 2026-10-28, before any data for these repositories was collected.

The evaluation records this file's SHA-256 in its manifest.

## Why

The first GitHub Actions test ([`phase1_v2_gha_tier_e_protocol.md`](phase1_v2_gha_tier_e_protocol.md))
found a result in the same direction and of similar size to Travis:

| | Within-repo AUC |
|---|---:|
| Tier C RF | 0.715 |
| Tier E blend | 0.770 |
| Δ | +0.055, positive in all 5 folds |

But only 3 of 5 fold CIs excluded 0, against the pre-registered 4. With 16–20
repositories per fold, the CIs were too wide, so the verdict was
**inconclusive**.

This test asks the same question on repositories and a time period that
nothing so far has touched.

## What is frozen

These are fixed exactly as in
[`phase1_v2_gha_tier_e_protocol.md`](phase1_v2_gha_tier_e_protocol.md) and the
code in `scripts/evaluate_gha_tier_e.py`:

- the unit, label, event filter and first-attempt filter;
- the 35-day rule for unknown completion times;
- the stream mapping, feature sets, models and LightGBM parameters;
- the walk-forward split (10 chronological folds, test folds 6–10);
- the metrics and the bootstrap.

The only change is the input file.

## Repository selection (fixed before collection)

1. **Candidate pool.** GitHub repository search for
   `stars:>10000 archived:false fork:false pushed:>2026-10-01`, sorted by
   stars, descending, first 300 results.
2. **Exclusions.**
   - The 30 repositories in the first GitHub Actions dataset.
   - The 4 finance-pilot repositories.
   - Repositories whose name contains "awesome", "interview", "roadmap",
     "tutorial", "course" or "list". These are curated lists or teaching
     material, not CI-tested software.
3. **Eligibility.** Measured with the Actions API `total_count` for runs created
   in the window (all attempts):
   - at least 500 `push` + `pull_request` runs that concluded `success` or
     `failure`;
   - at least 25 of those concluded `failure`.

   The API can't filter by `startup_failure`, so those rare runs don't count
   toward eligibility. They are still collected and labelled as in the first
   test.

   These counts decide eligibility only. They never enter a model.
4. **Selection.** Walk the candidates in star order and take the first 30
   eligible repositories.
5. **Cap.** Collect at most 4,000 qualifying runs per repository, taking the
   *earliest* in the window. The cap keeps API collection to a few hours and
   doesn't depend on outcomes.

## Window and data

- **Window:** runs created from 2026-08-15 00:00 UTC to 2026-09-30 23:59 UTC.
  This starts four months after the first dataset ends (2026-04-18), and every
  run has had four weeks to complete.
- **Runs:** from the Actions runs API. Fields:
  - `id`, `workflow_id`, `event`, `head_branch`, `head_sha`;
  - `run_attempt`, `conclusion`, `status`;
  - `created_at`, `updated_at`, `run_started_at`;
  - `actor.login`.
- **Commit details**, one API call per distinct `head_sha`, defined as in the
  first dataset:
  - `additions`, `deletions`;
  - `total_churn` = additions + deletions;
  - `files_modified` = number of changed files;
  - `msg_len` = length of the full commit message;
  - `num_parents`; `is_merge` = more than one parent;
  - `time_since_last_commit` = seconds since the repository's previous
    collected commit, by commit date.
- A run whose commit can't be fetched (deleted fork commit, API error after
  retries) is dropped. The count of dropped runs is reported.
- **Output:** written in the first dataset's column format, so
  `evaluate_gha_tier_e.py` runs unchanged on it.

## Primary result and decision rule

Unchanged from the first test.

- **Primary result:** the mean over folds 6–10 of macro within-repo AUC,
  Tier E blend minus Tier C RF, at a 0 h finish buffer.
- **Decision rule:**
  - "confirms" if the mean Δ > 0 and at least 4 of 5 fold CIs exclude 0;
  - "inconclusive" if the mean Δ > 0 otherwise;
  - "does not replicate" if Δ ≤ 0.

All secondary results are reported as in the first test: recall@15%, lift@15%,
pooled AUC, the 6 h buffer, new breakages, label validity, and the baselines.

**Secondary, decided now:** the same evaluation on the first dataset and the new
one combined (about 60 repositories). It is reported, but it cannot change the
primary verdict.

## Known limitations

- **Approximate commit timing.** `time_since_last_commit` is computed only from
  collected commits, as in the first dataset. In that dataset it matches
  "previous commit by commit date" on 78% of commits.
- **Eligibility counts include re-runs.** The counts use all attempts, while
  the model data keeps first attempts only.
- **Biased toward large projects.** Repository search favours popular, active
  projects.
