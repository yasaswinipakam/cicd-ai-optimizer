# Strict Pre-Build Baseline and Model Comparison

All rows use the frozen pipeline's availability-guarded `FEATURE_COLS`, 200,000-row cap, and deterministic 80/20 stratified split (`random_state=42`). These are research comparisons only; no row changes deployment selection.

| Model | Precision | Recall | F1 | AUC | TrainingSeconds | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Majority Class | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.00 | Always predicts the training-set majority class. |
| Previous-Status Heuristic | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.00 | No 'failed' category exists in this dataset's previous-status encoding; predicts passed. |
| Change-size Logistic Regression | 0.2394 | 0.2022 | 0.2193 | 0.5118 | 0.02 | Uses only strict derived change-size features. |
| Full Logistic Regression | 0.2467 | 0.5620 | 0.3428 | 0.5686 | 0.03 | Linear baseline over all strict pre-build features. |
| Random Forest | 0.3286 | 0.6217 | 0.4300 | 0.6902 | 1.71 | Same hyperparameters as the frozen production model; research-only refit. |
| XGBoost | 0.3066 | 0.6013 | 0.4061 | 0.6643 | 0.88 | Same hyperparameters as the prior XGBoost comparison; research-only refit. |
| Tabular MLP | 1.0000 | 0.0001 | 0.0002 | 0.5890 | 0.67 | Single-hidden-layer tabular comparison; no sequence data or tuning. |

Random Forest F1 is 0.4300; the change-size-only logistic baseline is 0.2193. The tabular MLP F1 is 0.0002.

The raw previous-status field has no `failed` category in the saved encoder (`build_found`, `merge_found`, `no_previous_build`), so the requested previous-build-failed heuristic cannot identify a prior failure and predicts passed for every row.
