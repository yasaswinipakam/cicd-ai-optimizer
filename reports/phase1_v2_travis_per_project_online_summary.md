# Travis per-project expanding online validation

Each project is ordered chronologically, split into ten contiguous folds, and evaluated on folds 6–10 after fitting on folds 1..j-1. RandomOverSampler is applied to the training fold only. The macro result averages completed projects equally; its 95% interval bootstraps projects, not builds.

Completed checkpoints: 75 folds from 15 projects. Complete five-fold projects included in the macro: 15.

Failure-class F1 must be read with prevalence; both are retained in the fold/project CSVs.

| metric          |   macro_mean |   repo_bootstrap_ci_low |   repo_bootstrap_ci_high |   complete_projects |
|:----------------|-------------:|------------------------:|-------------------------:|--------------------:|
| roc_auc         |       0.6419 |                  0.5970 |                   0.6869 |                  15 |
| pr_auc          |       0.3753 |                  0.2637 |                   0.5044 |                  15 |
| f1              |       0.2866 |                  0.1790 |                   0.4125 |                  15 |
| recall_at_10    |       0.1982 |                  0.1522 |                   0.2491 |                  15 |
| precision_at_10 |       0.3946 |                  0.2627 |                   0.5355 |                  15 |
| lift_at_10      |       1.9720 |                  1.5164 |                   2.4917 |                  15 |
| recall_at_15    |       0.2688 |                  0.2168 |                   0.3260 |                  15 |
| precision_at_15 |       0.3737 |                  0.2463 |                   0.5109 |                  15 |
| lift_at_15      |       1.7839 |                  1.4349 |                   2.1549 |                  15 |
| brier           |       0.1530 |                  0.1115 |                   0.1915 |                  15 |
| ece             |       0.1519 |                  0.1165 |                   0.1857 |                  15 |

Projects with fewer than five completed folds remain in the project CSV but are excluded from the macro until their remaining units are run or explicitly skipped.
