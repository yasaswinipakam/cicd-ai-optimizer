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

40 features (37 original + 3 streak/gap + 3 previously unused pre-build columns).
Hyperparameters tuned via `RandomizedSearchCV` (n_iter=25, 3-fold CV, F1 scoring) + LightGBM added.
Best RF: `max_depth=None, max_features=sqrt, class_weight=balanced, n_estimators=200`.
Best XGB: `n_estimators=400, max_depth=10, lr=0.2, subsample=0.9, reg_lambda=2.0`.
Best LGBM: `num_leaves=255, max_depth=10, lr=0.2, subsample=0.9, colsample_bytree=0.9`.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.8883 | 0.7397 | 0.7238 | **0.7317** | 0.9027 |
| Random Forest | 0.8938 | 0.8410 | 0.6112 | **0.7079** | 0.9077 |
| LightGBM | 0.8635 | 0.6528 | 0.7510 | 0.6984 | 0.8913 |
| Decision Tree | 0.7780 | 0.4822 | 0.7403 | 0.5840 | 0.8263 |
| SVM | 0.7587 | 0.4484 | 0.6374 | 0.5265 | 0.7760 |

No `tr_log_*` post-build field was used as a prediction feature.

## Authoritative result — full chronological split (this codebase, RF)

Trained on the earliest 80% of all 3.56M rows (2011–2016-03), tested on the
latest 20% (2016-03 to 2016-08).  Threshold selected on a chronological
validation window inside the training period.  **This is the honest headline
number.**

| Train rows | Test rows | Threshold | Precision | Recall | F1 | AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 2,849,851 | 712,463 | 0.30 | 0.3751 | 0.8142 | **0.5136** | **0.8218** |

Alert rate at this threshold: 42.5% of builds flagged; failure capture rate
(recall) 81.4%.  The threshold was chosen to maximise recall subject to a ≤40%
alert-rate ceiling and a precision floor equal to the base failure rate.

### Phase-1 rigor study cross-check

`src/models/phase1_rigor_evaluation.py` uses per-repository walk-forward folds
with block bootstrap CIs — a stricter within-repo protocol.  **Both are honest;
cite the one that matches your evaluation question.**

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
