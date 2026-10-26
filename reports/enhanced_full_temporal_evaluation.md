# Enhanced Full Temporal Evaluation

The deployable Random Forest uses the expanded availability-guarded pre-build feature schema. It was fitted on the earliest 80% of builds; its decision threshold was selected on a chronological validation window inside that training period, and the final 20% remained untouched until evaluation.

| Train rows | Test rows | Threshold | Precision | Recall | F1 | AUC | Alert rate | Failure capture rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,849,851 | 712,463 | 0.30 | 0.3751 | 0.8142 | 0.5136 | 0.8218 | 0.4254 | 0.8142 |

Train range: 2011-04-16 11:24:39+00:00 to 2016-03-07 14:53:32+00:00. Test range: 2016-03-07 14:53:32+00:00 to 2016-08-31 23:59:27+00:00. This result is a chronological within-dataset evaluation, not proof of cross-repository generalization.
