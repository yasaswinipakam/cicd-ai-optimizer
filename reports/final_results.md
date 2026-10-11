# Final Model Results

Evaluation was run on the reproducible 20% held-out split of the 200,000-row
training sample (`random_state=42`). Random Forest was selected as the best model
by F1 score.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.6511 | 0.3286 | **0.6217** | **0.4300** | **0.6902** |
| Decision Tree | 0.5748 | 0.2729 | 0.6064 | 0.3764 | 0.6170 |
| SVM | 0.5618 | 0.2579 | 0.5701 | 0.3551 | 0.5855 |
| XGBoost | 0.6278 | 0.3066 | 0.6013 | 0.4061 | 0.6643 |

These are the results of the one controlled retrain approved after the canonical
TravisTorrent file-count correction:
`gh_diff_files_added + gh_diff_files_deleted + gh_diff_files_modified`.
The feature schema, feature order, algorithms, hyperparameters, sample cap, and
80/20 stratified split all remained unchanged. No `tr_log_*` post-build field
was used as a prediction feature.

Compared with the prior honest Random Forest baseline (F1 **0.4182**), the
corrected mapping produces F1 **0.4300**, a change of **+0.0118**. Random Forest
remains the selected model by F1.

The exact machine-readable source is `artifacts/model_results.csv`; use that file
as the authoritative source when quoting numbers in the presentation.
