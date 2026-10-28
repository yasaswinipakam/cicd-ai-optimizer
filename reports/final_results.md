# Final Model Results

## Development reference (random-split, 200k rows) — NOT the headline result

These numbers are from a stratified random 80/20 split of a 200k-row sample
(`random_state=42`).  **They are a development reference only and must not be
cited as the model's real-world performance.**  A random split allows training
rows from the future to predict test rows from the past, which inflates all
metrics.  The authoritative evaluation protocol is the Phase-1 chronological
walk-forward (see below).

Three rolling history features are included: `repo_fail_rate_last5`,
`repo_fail_rate_last20`, and `repo_prior_fail_rate`. These are computed with the
finish-time-safe heap protocol (only builds that finished strictly before the
current build started), matching the Phase-1 rigor study.  A repository's
first builds have no completed history; their rates are filled with the
**training split's build-level failure rate** (0.200 on the full-mode training
window), saved with the model and reused at inference.  It replaced a
hardcoded 0.35 global constant.

39 features (20 direct pre-build + 19 derived).  Four columns are excluded and
blocked by the availability registry: `gh_num_pr_comments`,
`gh_num_issue_comments` and `gh_num_commit_comments` are undated snapshots that
can include comments written after the build, and `gh_num_commits_in_push` is
entirely missing in this export.  Removing them cost nothing measurable
(stacking AUC 0.912 → 0.912; deployed chronological AUC 0.8218 → 0.8208).

All artifacts are trained with the pinned environment in `requirements.txt` /
`constraints.txt` (Python 3.11, scikit-learn 1.9.1, pandas 3.1.0).

Hyperparameters tuned via `RandomizedSearchCV` (n_iter=25, 3-fold CV, F1 scoring).
Best RF: `max_depth=None, max_features=sqrt, class_weight=balanced, n_estimators=200`.
Best XGB: `n_estimators=400, max_depth=10, lr=0.2, subsample=0.9, reg_lambda=2.0`.
Best LGBM: `num_leaves=255, max_depth=10, lr=0.2, subsample=0.9, colsample_bytree=0.9`.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Stacking (RF+XGB+LGBM→LR) | 0.8990 | **0.8310** | 0.6531 | **0.7314** | **0.9116** |
| Random Forest | 0.8894 | 0.7490 | 0.7135 | 0.7309 | 0.9084 |
| XGBoost | 0.8895 | 0.7520 | 0.7086 | 0.7297 | 0.9003 |
| LightGBM | 0.8769 | 0.6990 | 0.7291 | 0.7138 | 0.8976 |
| Decision Tree | 0.7691 | 0.4693 | 0.7406 | 0.5745 | 0.8246 |
| SVM | 0.7583 | 0.4468 | 0.6229 | 0.5204 | 0.7723 |

Stacking ensemble (StackingClassifier, cv=5) has the highest AUC (0.9116) and
F1 (0.7314) at the default 0.5 threshold; RF and XGBoost are within 0.002 F1.  Use `--mode stack`
to train the stack; SHAP explanations on it use the XGB base estimator via
TreeExplainer.  Source: `artifacts/model_results.csv`.

No `tr_log_*` post-build field was used as a prediction feature.

## Chronological job-level result — full split (development reference, RF)

Trained on the earliest 80% of all 3.56M **job rows** (2011–2016-03), tested
on the latest 20% (2016-03 to 2016-08).  Threshold selected on a chronological
validation window inside the training period.

**NOT the Phase-1 headline.**  Phase-1 requires evaluation at **build level**
(~858k unique `tr_build_id` values) with per-repo walk-forward folds.  This
job-level result shows directional progress but cannot be directly compared to
the Phase-1 protocol or cited as a valid headline AUC.

| Train rows | Test rows | Threshold | Precision | Recall | F1 | AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 2,849,851 | 712,463 | 0.30 | 0.3712 | 0.8177 | **0.5106** | **0.8208** |

Alert rate at this threshold: 43.2% of builds flagged; failure capture rate
(recall) 81.8%.  The threshold was chosen to maximise recall subject to a ≤40%
alert-rate ceiling and a precision floor equal to the base failure rate.

### Phase-1 rigor study results (authoritative)

`src/models/phase1_rigor_evaluation.py` uses per-repository walk-forward folds
with block bootstrap CIs — a stricter within-repo protocol at the correct
**build level** (~858k builds).

**These two results are NOT directly comparable** and should not both be
labelled "honest headlines":
- The full-mode result above operates at **job level** (3.56M rows).  A single
  build can span multiple job rows, so label/feature duplication inflates
  apparent sample size and can place related jobs across the train/test boundary.
- The Phase-1 within-repo result operates at **build level** with per-repo
  walk-forward folds — this is the agreed Phase-1 unit of analysis.

Cite the Phase-1 build-level result as the headline.  The job-level full-mode
result is a development reference that shows directional improvement only.

#### Headline: Tier E + LightGBM rank/binary blend (confirmation folds 7–10)

Tier E adds finish-time-safe **stream** history (a stream is one PR, or one
branch for push builds), **in-flight** context (is the previous build still
running, queued builds, time since the stream last triggered) and build-matrix
size to Tier C.  The model is an equal blend of a LightGBM `rank_xendcg` ranker
(queries = repository × calendar month, so it optimises *within-repo* ordering)
and a binary LightGBM, mapped to probabilities by one global Platt fit on
training scores.

**Protocol.** Every design choice was made on fold 6 only and frozen in code
(`tier_e_protocol` in `phase1_v2_run_manifest.json`, which lists every variant
tried).  Folds 7–10 were scored once, afterwards.  A brute-force recomputation
(`_assert_stream_history`) re-checks the new history values on 40 repositories
in every run; the smoke test confirms it catches a deliberately leaked feature.
A global train-label shuffle drops the ranker to 0.48–0.50 within-repo AUC.

| Finish buffer | Tier C RF | **Tier E blend** | Δ (paired, every fold's 95% CI > 0) |
|---:|---:|---:|---:|
| 0h | 0.628 | **0.698** | **+0.070** |
| 1h | 0.600 | **0.681** | +0.081 |
| 6h | 0.585 | **0.672** | +0.087 |

Macro within-repo AUC, mean of folds 7–10.  Per-fold 0h: 0.702 / 0.713 / 0.683 / 0.694.
Recall@15% within repo improves by +0.068 at 0h.  The gain *grows* under
stricter buffers, because the in-flight features stay informative when
outcomes are withheld longer.

Attribution (folds 7–10, 0h, within-repo AUC):

| Variant | Within-repo AUC | Pooled AUC |
|---|---:|---:|
| Tier C RF (previous headline) | 0.628 | 0.856 |
| Tier C, binary LightGBM | 0.633 | — |
| Tier C, LightGBM ranker | 0.664 | — |
| Tier E, binary LightGBM | 0.652 | **0.881** |
| Tier E, LightGBM ranker | 0.696 | — |
| **Tier E, blend** | **0.698** | 0.828 |
| Tier E blend without job-count features | 0.695 | 0.826 |

- The **ranking objective** is the single largest lever (+0.031 on Tier C alone).
- The **stream/in-flight features** add ~+0.03 on top of it.
- Job-count features contribute only ~+0.003, so the result does not depend on them.
- **Trade-off:** the blend gives up 0.028 pooled AUC versus Tier C RF (PR-AUC is
  slightly higher, 0.667 vs 0.660).  If pooled ranking across repositories is the
  operating goal, use **Tier E binary LightGBM**, which beats Tier C RF on both
  metrics (within-repo +0.024, pooled +0.025).

Sources: `reports/phase1_v2_walkforward_tier_e_vs_c.csv` (paired repository
bootstrap), `reports/phase1_v2_walkforward_tier_e_summary.csv`,
`scripts/compare_phase1_walkforward_tier_e.py`.

### New breakages only (post-hoc rescoring of the saved predictions)

Much of the headline comes from builds on a branch or PR that is already red.
In folds 7–10, **55% of failures** occur on streams whose last known outcome was
a failure (64% of those builds fail).  Only **25% of failures** are new breakages,
meaning the immediately preceding build on the same stream passed.  Those
builds fail at just 6.6%.  To check that the model does more than persistence,
the saved fold predictions were rescored on that subset.  Nothing was refit,
and the subset is defined by the outcome of the previous build, which is never
used as a feature.

| Finish buffer | Repo last-5 rate (persistence) | Tier C RF | **Tier E blend** | Δ E − C |
|---:|---:|---:|---:|---:|
| 0h | 0.504 | 0.556 | **0.622** | +0.066 |
| 1h | — | 0.540 | **0.600** | +0.060 |
| 6h | — | 0.535 | **0.601** | +0.066 |

Macro within-repo AUC on new-breakage builds, mean of folds 7–10, with about
420–450 repositories per fold.  Per-fold 0h Tier E: 0.637 / 0.625 / 0.618 / 0.607,
each with a repository-bootstrap 95% CI of about ±0.02.  The E − C delta's CI is
above 0 in every fold at every buffer; the smallest lower bound is +0.026.

At 0h, with a 15% budget taken within the new-breakage builds, Tier E catches
30% of new breakages (lift **1.78×**).  Tier C RF catches 24% (1.40×).

- **The honest headline for fresh breakages is ~0.62, not 0.70.**  The
  0.698 figure stands as an all-builds number, but it partly reflects
  "already failing" streams.
- **Persistence explains none of the new-breakage signal.**  The
  repository last-5 rate is at chance (0.504) on this subset.  The remaining
  0.62 comes from change, stream and in-flight context.
- **Tier E's advantage over Tier C is not a persistence artifact.**  It holds
  undiminished on new breakages (+0.066 vs +0.070 on all builds).
- On already-failing streams, Tier C RF ranks *below chance* (0.484) at
  telling which builds stay red.  Tier E manages 0.555 at 0h and 0.618 at 6h.
- Builds with no completed stream history make up 18% of failures.  Tier E
  scores 0.667 on them.

This analysis is post hoc.  Folds 7–10 had already been scored once for the
Tier E headline.

Sources: `reports/phase1_v2_new_breakage.csv` (per fold, model and stratum,
with CIs), `reports/phase1_v2_new_breakage_vs_c.csv` (paired deltas),
`scripts/analyze_phase1_new_breakage.py`.

### External test on GitHub Actions (pre-registered)

The frozen recipe was rerun on fresh data: 67,980 first-attempt GitHub Actions
`push`/`pull_request` runs from 30 repositories, covering 2025–26. The protocol
was fixed before any fit.

| Within-repo AUC, folds 6–10, 0h | Tier C RF | Tier E blend | Δ |
|---|---:|---:|---:|
| All runs | 0.715 | 0.770 | +0.055 |

- **Within-repo AUC:** Δ is positive in all 5 folds, but only 3 of 5 fold CIs
  exclude 0. The protocol required 4, so the verdict is **inconclusive**.
- **Recall@15%:** 0.48 vs 0.37, with the CI excluding 0 in 4 of 5 folds.
- **6 h finish buffer:** the gain shrinks to +0.021.

Details: [`phase1_v2_gha_tier_e_results.md`](phase1_v2_gha_tier_e_results.md).

#### Previous baseline: walk-forward (folds 6–10, 0h finish buffer, Tier C RF)

| Fold | Repos | Macro within-repo AUC | Recall@15% |
|---:|---:|---:|---:|
| 6 | 525 | 0.643 | 0.329 |
| 7 | 530 | 0.634 | 0.308 |
| 8 (predeclared) | 544 | 0.646 | 0.308 |
| 9 | 511 | 0.613 | 0.300 |
| 10 | 511 | 0.621 | 0.279 |
| **avg** | | **0.631** | **0.305** |

History features (Tier C) add **+0.102 macro AUC** over no-history (Tier A),
95% CI all-positive across all folds.

#### Development test split (fixed 80/20, Tier C RF)

| Finish buffer | Repos (both classes) | Macro within-repo AUC | Pooled AUC |
|---:|---:|---:|---:|
| 0h | 13 | 0.678 | **0.824** |
| 1h | 13 | 0.612 | 0.796 |

#### Finish-buffer sensitivity (walk-forward, Tier C)

Adding a finish buffer guards against near-simultaneous builds leaking outcomes.
Cost paid in AUC (averaged over folds 6–10):

| Buffer | Mean ΔAUC vs 0h | 95% CI |
|---:|---:|---:|
| 1h | −0.030 | all negative |
| 6h | −0.046 | all negative |

At 1h buffer the headline AUC drops from ~0.631 to ~0.601; at 6h to ~0.585.
Cite the 0h result as headline; sensitivity numbers quantify how much stricter
assumptions cost.

#### Negative result: Cplus (streak/gap features) hurts AUC

Adding `HISTORY_PLUS` (hist_failure_streak, hist_hours_since_last_build,
hist_hours_since_last_failure, hist_repo_fail_rate_10/20) to Tier C **reduces**
macro within-repo AUC by −0.019 on average (4 of 5 folds have CI entirely below 0).

| Fold | Tier C AUC | Tier Cplus AUC | Δ (Cplus − C) |
|---:|---:|---:|---:|
| 6 | 0.643 | 0.633 | −0.010 |
| 7 | 0.634 | 0.612 | **−0.022** |
| 8 | 0.646 | 0.633 | **−0.013** |
| 9 | 0.613 | 0.595 | **−0.018** |
| 10 | 0.621 | 0.590 | **−0.030** |

Bold = CI entirely below 0 (statistically significant degradation).

The streak/gap features added to `engineer.py` (application training) improve the
app model's dev-split AUC but degrade the within-repo Phase-1 metric — likely
because they introduce overfitting on short per-repo histories.  Repo-level
streaks mix all branches and PRs together; Tier E's per-stream history is
the version of this idea that does help.

See `reports/phase1_v2_travis_per_project_online_summary.md` for per-project
online validation: macro AUC 0.642, 95% CI [0.597, 0.687] across 15 projects.

Run again with:

```
python -m src.models.train --mode full
```

## Historical comparison (13 features — pre-build change-size and temporal only, random split)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.6511 | 0.3286 | 0.6217 | 0.4300 | 0.6902 |
| Decision Tree | 0.5748 | 0.2729 | 0.6064 | 0.3764 | 0.6170 |
| SVM | 0.5618 | 0.2579 | 0.5701 | 0.3551 | 0.5855 |
| XGBoost | 0.6278 | 0.3066 | 0.6013 | 0.4061 | 0.6643 |

The exact machine-readable source is `artifacts/model_results.csv`; use that file
as the authoritative source when quoting numbers in the presentation.
