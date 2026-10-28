# Confirmation test of the frozen Tier E recipe on new GitHub Actions repositories — results

The protocol is [`phase1_v2_gha_confirmation_protocol.md`](phase1_v2_gha_confirmation_protocol.md).
It was written and hashed before any run was collected: SHA-256 `e9bcd52d…8cdb`,
recorded at 2026-10-28T02:54:31Z in `phase1_v2_gha_confirmation_protocol_hash.json`
and again in `phase1_v2_gha_confirmation_manifest.json`.

- **Collection:** `scripts/collect_gha_confirmation.py`.
- **Evaluation:** `scripts/evaluate_gha_tier_e.py`, unchanged apart from the
  input file and one date-parsing fix (see *Disclosed deviations*).

## Data

**Repositories.** 30 repositories, none used before.

- They are the first 30 eligible candidates from the star-sorted search. The
  30th was candidate #63.
- The full list, with exclusions, is in
  [`phase1_v2_gha_confirmation_candidates.csv`](phase1_v2_gha_confirmation_candidates.csv).

**Window.** Runs created from 2026-08-15 to 2026-09-30.

**Runs.**

- 82,349 qualifying first-attempt `push`/`pull_request` runs after the
  4,000-per-repository cap.
- 3,773 runs (4.6%) were dropped because their commit could not be fetched. The
  API returned 422 ("no commit found") for 1,123 SHAs, almost all on fork pull
  requests.
- The drops are concentrated in a few repositories:

  | Repository | Runs dropped | Share of its runs |
  |---|---:|---:|
  | openclaw | 1,770 | 44% |
  | hermes-agent | 1,380 | 35% |
  | opencode | 481 | |
  | graphify | 122 | |
  | n8n | 20 | |

- Dropped runs failed more often than kept ones (38.5% vs 28.2%).

**Modelled data.** 78,576 runs; 28.2% failed (the first test: 8.9%).

- No run exceeded the 35-day completion rule.
- 28 runs are never-executed proxies.

**Split.** 10 chronological folds of about 7,860 runs each; test folds 6–10.

- 18–24 repositories have both outcomes in each test fold, and 24 appear in at
  least one.
- The 4,000-run cap took each busy repository's *earliest* runs. Six
  repositories therefore reached the cap within the first days of the window and
  appear only in training folds: openclaw, hermes-agent, n8n, opencode, dify and
  AutoGPT. This follows from the pre-registered cap. It does not depend on
  outcomes.

## Pre-registered verdict: confirms

| Within-repo AUC, mean of folds 6–10 (0h) | Confirmation | First GHA test | Travis (folds 7–10) |
|---|---:|---:|---:|
| Tier C RF | 0.763 | 0.715 | 0.628 |
| **Tier E blend** | **0.911** | 0.770 | 0.695 |
| Δ E − C | **+0.149** | +0.055 | +0.067 |
| Fold CIs above 0 | **5 of 5** | 3 of 5 | 4 of 4 |

- **Per-fold Δ:** +0.135 / +0.149 / +0.188 / +0.140 / +0.132.
- **Lowest CI bound:** +0.090. Every fold clears zero by a wide margin.
- **The rule:** "confirms" if the mean Δ > 0 and at least 4 of 5 fold CIs
  exclude 0. Both conditions hold. The script prints this as `"transfers"`, the
  first protocol's word for the same outcome.

**Recall and lift.**

- **Recall@15%:** Tier E catches 43% of failures in a 15% within-repo review
  budget, against 31% for Tier C. The delta's CI excludes 0 in 5 of 5 folds.
- **Lift@15%:** 2.82× vs 2.04×.
- **Pooled AUC:** 0.940 vs 0.820.

## Secondary results

| Setting | Tier C RF | Tier E blend | Δ | Fold CIs > 0 |
|---|---:|---:|---:|---:|
| 0h, all runs (primary) | 0.763 | 0.911 | +0.149 | 5/5 |
| 0h, label-validity sensitivity | 0.765 | 0.911 | +0.147 | 5/5 |
| 0h, new breakages only | 0.642 | 0.683 | +0.041 | 2/5 |
| 6h buffer, all runs | 0.741 | 0.846 | +0.105 | 5/5 |
| 6h buffer, new breakages | 0.637 | 0.634 | −0.004 | 0/5 |

**Baselines, 0h:**

| Variant | Within-repo AUC |
|---|---:|
| `A_rf` (no history) | 0.746 |
| `C_lgbmblend` (ranking objective, no stream features) | 0.805 |
| `E_lgbm` (stream features, binary only) | 0.912 |
| Last-5 repository failure rate | 0.570 |
| Repository prior | 0.500 |

**What the secondary results show:**

- **The 6 h buffer keeps most of the gain here:** +0.105, 5 of 5 folds. In the
  first test it fell to +0.021.
- **Label validity doesn't matter.** Only 28 never-executed proxies exist.
- **New breakages are the weak spot.** These are runs whose previous run on the
  same workflow and stream passed. There are only about 91 such failures per
  fold, so the CIs are wide.
  - At 0h, Tier E is ahead by +0.041; 2 of 5 fold CIs exclude 0.
  - With a 6 h buffer the two models tie.
  - In plain words, most of Tier E's gain comes from knowing how *this* stream
    has been doing. It adds little for the first failure after a pass.
- **Attribution matches the first GHA test, not Travis.**
  - The ranking objective adds +0.042: `C_lgbmblend` vs `C_rf`.
  - The stream and in-flight features add the rest: `E_lgbm` scores 0.912,
    against 0.805 for `C_lgbmblend`.

## Exploratory check: is this only "it failed last time"? (not pre-registered)

The gain is almost three times the first test's, and failures are three times as
common. So I checked whether a naive persistence rule explains it. The rule
scores each run by whether the previous finished run on its stream failed, with
ties broken by the stream's last-5 failure rate.

Code: `scripts/analyze_gha_persistence.py`, run on the saved fold predictions.

| Fold | Persistence | Tier C RF | Tier E blend |
|---|---:|---:|---:|
| 6 | 0.710 | 0.779 | 0.914 |
| 7 | 0.720 | 0.749 | 0.898 |
| 8 | 0.727 | 0.727 | 0.915 |
| 9 | 0.729 | 0.777 | 0.917 |
| 10 | 0.718 | 0.781 | 0.913 |

- **Persistence alone scores about 0.72**, below Tier C. Tier E is not simply
  replaying the previous outcome.
- **Tier E beats Tier C in all 24 test repositories.** The mean per-repository
  Δ ranges from +0.021 (MoneyPrinterTurbo) to +0.329 (PowerToys).
- **The gain does not track a repository's failure rate** (correlation 0.03).
  The high-failure repositories, such as markitdown at 86% failures and the
  public-apis list repository at 78%, are not what drives it.
- **Only about half the test runs have any earlier finished run on their
  stream** (49–56% per fold).

## Combined analysis (secondary, pre-registered)

The same evaluation on both GHA datasets together: 146,556 runs from 60
repositories, 19.2% of them failures. The protocol says this is reported but
cannot change the primary verdict.

| Setting | Tier C RF | Tier E blend | Δ | Fold CIs > 0 |
|---|---:|---:|---:|---:|
| 0h, all runs | 0.685 | 0.864 | +0.179 | 5/5 |
| 0h, label-validity sensitivity | 0.680 | 0.866 | +0.186 | 5/5 |
| 0h, new breakages only | 0.586 | 0.673 | +0.087 | 2/5 |
| 6h buffer, all runs | 0.673 | 0.789 | +0.116 | 5/5 |
| 6h buffer, new breakages | 0.582 | 0.634 | +0.052 | 0/5 |

- **Per-fold Δ:** +0.150 / +0.207 / +0.162 / +0.221 / +0.153. The lowest CI
  bound is +0.092.
- **Recall@15%:** 0.40 vs 0.25.
- **How to read it:** the folds are chronological over both periods.
  - The first dataset spans 2025 to April 2026 and fills roughly the first 4.6
    folds.
  - So test folds 6–10 are almost entirely the confirmation period, and the
    models are trained on both.
- **Tier C transfers worse across that gap than Tier E.** Tier C's
  within-repo AUC falls from 0.763 (confirmation alone) to 0.685. Tier E's falls
  less, from 0.911 to 0.864.

## Disclosed deviations

**Date parsing (affects only the combined analysis).**

- The new dataset writes `updated_at` as `…Z`, while the first dataset writes
  `+00:00`.
- When the two files were concatenated, pandas inferred one format and silently
  coerced every value in the other style to NaT. The run-filter then dropped all
  new rows.
- So the first combined run reproduced the first test exactly: 67,980 runs, 30
  repositories. I caught this because the counts matched.
- **Fix:** `load_runs` now parses with an explicit `format="ISO8601"`.
- **Checks:**
  - On the first dataset, the parsed `created_at` and `updated_at` columns are
    identical to before, with the same NaT counts. So the first test's results
    are unaffected.
  - The confirmation run reads one file, and all 78,576 of its rows parsed.
- The faulty combined run's log is kept at
  `runs/logs/gha_combined_eval_dropped_new_rows.log`.

**Eligibility screen.**

- The name filter did not catch `public-apis/public-apis`. It is a list
  repository whose CI checks links, but its name has none of the excluded words.
  It stays in, as the protocol requires.
- It has 2,114 runs with a 78% failure rate. Its per-repository Δ is +0.046,
  below the median, so it does not inflate the result.

## Reading

- **On new repositories and a new period, the frozen recipe clears the
  pre-registered bar easily.** The confirmation gives +0.149 within-repo AUC,
  with 5 of 5 fold CIs above 0 and recall@15% up from 31% to 43%. This settles
  the first test's "inconclusive": the gain replicates.
- **The size is not stable across datasets.**

  | Dataset | Gain |
  |---|---:|
  | Travis | +0.067 |
  | First GHA test | +0.055 |
  | Confirmation | +0.149 |

  These repositories fail far more often, and their CI streams are more
  informative. The honest claim is "Tier E beats Tier C consistently". The size
  of the gain depends on the project.
- **The gain is about stream context, not first failures.** Tier E's lead on new
  breakages is small and disappears with a 6 h delay. So it ranks builds best
  when the stream's recent outcomes are known.
- **Selection limits.**
  - The repositories are popular and very active.
  - Six of the busiest are training-only, because of the earliest-runs cap.
  - 4.6% of runs were lost to unfetchable fork commits.
