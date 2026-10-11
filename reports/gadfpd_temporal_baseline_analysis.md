# GADFPD Global Temporal Baseline

This is a separate GitHub Actions research experiment. It uses only the 16 documented pre-run fields from the modern dataset; repository, workflow identifier, conclusion, duration, status, and post-run timestamps are excluded from the feature matrix. Models train on older rows and test only on the later global 20%. The 0.50 decision threshold is a baseline, not an operational alert recommendation.

| Model               |   TrainRows |   TestRows |   TestFailureRate |   Precision |   Recall |     F1 |    AUC |   AlertRate |   TrainingSeconds |
|:--------------------|------------:|-----------:|------------------:|------------:|---------:|-------:|-------:|------------:|------------------:|
| majority            |      116888 |      29222 |            0.0468 |      0.0000 |   0.0000 | 0.0000 | 0.5000 |      0.0000 |            0.1468 |
| logistic_regression |      116888 |      29222 |            0.0468 |      0.0760 |   0.5724 | 0.1342 | 0.6634 |      0.3526 |            2.0086 |
| random_forest       |      116888 |      29222 |            0.0468 |      0.1164 |   0.6433 | 0.1972 | 0.8023 |      0.2586 |            1.0497 |

Do not compare these scores directly with the deployed TravisTorrent model: platform, era, label prevalence, and features differ.
