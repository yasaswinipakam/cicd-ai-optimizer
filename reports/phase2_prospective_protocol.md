# Forward-looking test of the deployed model — protocol

Written 2026-10-28, before any run from the test window was collected or
looked at. `phase2_prospective_protocol_hash.json` records this file's SHA-256
and the fingerprints of everything frozen below.

## Question

Every earlier result tests the Tier E *recipe* on data that already existed.
This test asks whether the *deployed system* holds up on runs that did not exist
when it was frozen:

- the model file;
- its history features;
- its per-repository alert rule.

## What is frozen

**The model.** `artifacts/tier_e_gha_blend.joblib`, SHA-256
`57c26f170b6dae23c92a268c672663fc5243dc049aa8135374fdf3fabcb12860`.

- Trained on runs created through 2026-09-30.
- Built by `src/models/train_tier_e_gha.py` at commit `7959104`.
- It is not retrained, recalibrated or changed in any way.

**The code** at commit `7959104`; the hash file records each file's SHA-256:

| File | Role |
|---|---|
| `src/models/tier_e.py` | the model class |
| `src/inference/alert_policy.py` | the alert rule: riskiest 15% of a repository's last 100 scored runs, after 20 scored runs; before that, the model's typical-repository threshold, 0.893 |
| `scripts/evaluate_gha_tier_e.py` | `load_runs` and `with_history`, at a 0 h finish buffer: the unit, label, filters and history features |
| `scripts/compare_phase1_walkforward_tiers.py` | the per-repository metrics and the bootstrap |

## Data

**Repositories.** The 60 repositories the model was trained on, listed in
[`phase2_prospective_repositories.csv`](phase2_prospective_repositories.csv):

- the 30 from the first GitHub Actions test;
- the 30 from the confirmation test.

This is a test of the *same repositories in a later period*, not of new
repositories.

**Window and collection.**

- **Window:** runs created from 2026-10-01 00:00 UTC to 2026-10-31 23:59:59
  UTC.
- **Collection:** starts no earlier than 2026-11-28, so every run has had four
  weeks to finish.
- **Method:** as for the confirmation dataset, with
  `scripts/collect_gha_confirmation.py`'s run-listing, commit-fetch and build
  steps. Only the repository list and the window change.
  - First-attempt `push` and `pull_request` runs.
  - The earliest 4,000 qualifying runs per repository.
  - A run whose commit cannot be fetched is dropped, and the drops are
    reported.
- **Labels:** `success` is a pass; `failure` and `startup_failure` are
  failures. Runs with unknown completion over 35 days are dropped, as before.

**History.** Features come from the October runs alone, exactly as if the app
had been connected on 2026-10-01 with an empty database.

**Warm-up.**

- Each repository's first 25% of collected runs, in creation order, is warm-up.
  These runs are scored, and they feed history and the alert rule, but they
  are not evaluated.
- The remaining 75% are the evaluated runs.
- A share rather than a number of days keeps busy repositories, which reach
  4,000 runs within days, in the test.

**Evaluated repositories.** A repository counts in the primary result if its
evaluated runs include at least 10 failures and at least 10 passes. The number
excluded is reported.

## Scores

- **Deployed:** the frozen model's probability for every run, in creation
  order.
- **Persistence baseline:** whether the previous finished run on the same
  stream failed, with ties broken by 0.01 × the stream's last-5 failure rate.
  Without stream history, it uses the pooled failure rate of all warm-up runs.

## Primary result and decision rule

**Primary result.**

- The metric is macro within-repository AUC on evaluated runs.
- It is computed for the deployed score minus the persistence baseline (Δ),
  and for the deployed score alone.
- The 95% CI comes from a repository bootstrap: 1,000 resamples, seed
  20261013.

**Decision:**

- **"Holds up":** the lower CI bound of Δ is above 0, *and* the deployed
  macro AUC is at least 0.80.
- **"Degrades":** the lower bound of Δ is above 0, but the deployed AUC is
  below 0.80.
- **"Does not beat persistence":** otherwise.

**Why 0.80.** The holdout gave 0.913. A drop of more than about 0.1 would mean
the model no longer works as validated, even if it still beats persistence.

## Secondary results, reported, not decisive

- **The alert rule as it would have run live:** each repository's share of
  runs flagged (range), mean share of a repository's failures caught, and
  pooled precision.
- **Other metrics:** within-repository recall at a hindsight 15% budget, and
  pooled AUC.
- **New breakages:** the same comparison on runs whose previous run of that
  workflow on that stream passed.
- **By source:** results for the first-test and confirmation repositories
  separately.

## Commitments

- **The evaluation script** (`scripts/evaluate_prospective.py`) will be
  written and committed before collection starts. It may only implement this
  protocol.
- **Deviations:** any deviation forced by the data (for example, an API change)
  is disclosed in the results with its reason.
- **Reporting:** the result is reported whatever it is.

## Known limitations

- **Same repositories.** This tests a later period, not repositories the model
  has never seen.
- **Commit timing.** `time_since_last_commit` comes only from collected
  commits, as in training.
- **The cap.** It keeps only each busy repository's first days of October.
- **Drift.** Repositories may change CI setups or go quiet in October. That is
  part of what a forward-looking test measures.
