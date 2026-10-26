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

### Phase-1 rigor study cross-check

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

| Metric | Phase-1 Tier A (no history) | Phase-1 Tier C (+ history) | This codebase (full mode) |
|---|---:|---:|---:|
| Pooled / global AUC | ~0.72 | ~0.76 | **0.822** |
| Macro within-repo AUC | ~0.64 | — | — |
| F1 (chosen threshold) | ~0.43 | ~0.48 | **0.514** |

See `reports/phase1_v2_travis_per_project_online_summary.md` for per-project
details (macro AUC 0.64, 95% CI [0.60, 0.69]).

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
