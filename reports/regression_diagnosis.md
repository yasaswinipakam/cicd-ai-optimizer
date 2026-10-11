# Random Forest F1 Regression Diagnosis

## Scope

This is a read-only investigation of the difference between the notebook's
reported Random Forest F1 of **0.755** and the leak-free pipeline's F1. The
initial read-only diagnosis used the then-saved model (F1 **0.4182**). The
canonical TravisTorrent file-count schema fix was then separately approved, and
one controlled train/evaluate/anomaly-refresh cycle produced the current
saved-model result (F1 **0.4300**). No hyperparameters were changed.

## Current saved-model evaluation

The exact deterministic 200,000-row sample and 80/20 stratified split used by
the current training/evaluation pipeline were recreated.

| Measure | Value |
| --- | ---: |
| Confusion matrix `[[TN, FP], [FN, TP]]` | `[[20782, 10753], [3202, 5263]]` |
| Precision (failed) | 0.3286 |
| Recall (failed) | 0.6217 |
| F1 (failed) | 0.4300 |

The model is not simply predicting every build as passed: it identifies 5,263
of 8,465 failed builds (61.22% recall), but has many false positives, which
limits precision to 31.76%.

## Encoder fallback audit

The transform-only encoder path substitutes a fallback only when a source
category is absent from the fitted encoder's classes. The test set had **no
such substitutions**.

| Encoded feature | Fallback / unseen rows | True unseen rate | Missing source rows |
| --- | ---: | ---: | ---: |
| `prev_status_enc` | 0 / 40,000 | 0.0000% | 0 / 40,000 |
| `lang_enc` | 0 / 40,000 | 0.0000% | 0 / 40,000 |

Therefore the encoder-refitting/fallback hypothesis is not supported. Both
encoded features also remain informative in the saved model (`lang_enc` is
fifth and `prev_status_enc` sixth by feature importance), rather than having
dropped out of the ranking.

## Current Random Forest feature importance

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `total_files_changed` | 0.216388 |
| 2 | `hour` | 0.216029 |
| 3 | `month` | 0.195130 |
| 4 | `day_of_week` | 0.103727 |
| 5 | `lang_enc` | 0.101290 |
| 6 | `prev_status_enc` | 0.051852 |
| 7 | `is_master` | 0.033457 |
| 8 | `is_pr` | 0.031624 |

The canonically corrected `total_files_changed` feature is now non-zero and
ranks first; this confirms the intended real dataset information reaches the
model. The overall F1 remains in the same leak-free range.

## Controlled comparisons

The first comparison deliberately reproduced the notebook-style preprocessing
condition: fit both LabelEncoders and the median imputer on the full dataset
before splitting, then fit the Random Forest with the *same* specified
hyperparameters.

| Comparison | Random Forest F1 |
| --- | ---: |
| Prior honest baseline before canonical schema correction | 0.4182 |
| Current saved leak-free pipeline after canonical schema correction | 0.4300 |
| Full-dataset encoder + imputer before split | 0.4154 |
| Diagnostic-only model including post-build `tr_log_*` fields | 0.6737 |

Fitting encoders/imputation on the full data does **not** recover the notebook
score. In contrast, admitting fields written after build execution creates a
large 0.2583 absolute F1 increase. Those `tr_log_*` fields are intentionally
excluded from the pipeline because they reveal information only available after
the outcome is known.

The diagnostic log-field model does not reproduce exactly 0.755, which is
expected because the original notebook cell and its exact selected columns are
not available in this repository. It is nevertheless direct evidence that
post-build leakage is capable of producing the reported performance gap, while
the proposed encoder cause is not.

## Verdict

**Hypothesis A confirmed — 0.755 was leaked, current pipeline is correct.**

The current 0.4300 F1 is the defensible leak-free estimate for the specified
pre-build feature schema. Do not present 0.755 as a deployable prediction
metric. The viva/report should state that the earlier notebook result used
post-build information or an equivalent non-deployable feature setup, whereas
the production pipeline prohibits `tr_log_*` prediction inputs.

## Canonical file-count schema fix and controlled retrain

The official TravisTorrent schema defines the total changed-file count as
`gh_diff_files_added + gh_diff_files_deleted + gh_diff_files_modified`.
`gh_diff_tests_*` are test-case counts and `gh_diff_src_files` is only a
source-file subset. Source code now uses the canonical three-field total, with
legacy `git_diff_*_files` accepted only for older API clients. The approved
single train/evaluate/anomaly run completed using this mapping. The 0.0118 F1
increase over the honest 0.4182 baseline is reported as a schema-correction
effect, not as a comparison with the leakage-affected notebook score.

## Reproduction

Run:

```bash
.venv/bin/python -m src.models.diagnose_regression
```

The scratch script prints the same read-only diagnostic and does not write
production artifacts.
