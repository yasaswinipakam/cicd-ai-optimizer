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
current build started), matching the Phase-1 rigor study.  Cold-start default
is 0.0.

43 features (37 original + 3 streak/gap + 3 previously unused pre-build columns).
Hyperparameters tuned via `RandomizedSearchCV` (n_iter=25, 3-fold CV, F1 scoring) + LightGBM added.
Best RF: `max_depth=None, max_features=sqrt, class_weight=balanced, n_estimators=200`.
Best XGB: `n_estimators=400, max_depth=10, lr=0.2, subsample=0.9, reg_lambda=2.0`.
Best LGBM: `num_leaves=255, max_depth=10, lr=0.2, subsample=0.9, colsample_bytree=0.9`.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Stacking (RF+XGB+LGBM→LR) | 0.8981 | **0.8378** | 0.6394 | 0.7253 | **0.9121** |
| XGBoost | 0.8883 | 0.7397 | 0.7238 | **0.7317** | 0.9027 |
| Random Forest | 0.8938 | 0.8410 | 0.6112 | 0.6796 | 0.8940 |
| LightGBM | 0.8635 | 0.6528 | 0.7510 | 0.6984 | 0.8913 |
| Decision Tree | 0.7780 | 0.4822 | 0.7403 | 0.5840 | 0.8263 |
| SVM | 0.7587 | 0.4484 | 0.6374 | 0.5265 | 0.7760 |

Stacking ensemble (StackingClassifier, cv=5) has the highest AUC (0.9121) and
precision (0.8378), meaning fewer false alarms.  XGBoost leads on F1 (0.7317)
at the default 0.5 threshold; threshold tuning on the stacking model would likely
close that gap.  Use `--mode stack` to train it; SHAP explanations on the stack
use the XGB base estimator via TreeExplainer.

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
| 2,849,851 | 712,463 | 0.30 | 0.3751 | 0.8142 | **0.5136** | **0.8218** |

Alert rate at this threshold: 42.5% of builds flagged; failure capture rate
(recall) 81.4%.  The threshold was chosen to maximise recall subject to a ≤40%
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

#### Walk-forward (folds 6–10, 0h finish buffer, Tier C RF)

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
because they introduce overfitting on short per-repo histories.  **The Phase-1
headline remains Tier C (no streak features).**

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
