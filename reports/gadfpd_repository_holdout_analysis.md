# GADFPD Temporal Unseen-Repository Evaluation

Each row holds out one entire repository from training and tests only its later builds after the global temporal cutoff. The five largest repositories satisfying at least 200 later builds and 15 later failures were selected before fitting; this is a bounded, transparent study rather than performance cherry-picking.

| Model         |   TrainRows |   TestRows |   TestFailureRate |   Precision |   Recall |     F1 |    AUC |   AlertRate |   TrainingSeconds | Repository            | TrainEnd                  | TestStart                 |
|:--------------|------------:|-----------:|------------------:|------------:|---------:|-------:|-------:|------------:|------------------:|:----------------------|:--------------------------|:--------------------------|
| random_forest |      116888 |       4687 |            0.0911 |      0.1203 |   0.7775 | 0.2084 | 0.6456 |      0.5886 |            0.9749 | microsoft/vscode      | 2026-04-05 11:36:49+00:00 | 2026-04-08 07:20:46+00:00 |
| random_forest |      112528 |       4522 |            0.0064 |      0.0000 |   0.0000 | 0.0000 | 0.4119 |      0.2479 |            0.8944 | elastic/elasticsearch | 2026-04-05 11:36:49+00:00 | 2026-04-05 13:12:09+00:00 |
| random_forest |      111530 |       3541 |            0.0192 |      0.0938 |   0.5294 | 0.1593 | 0.9184 |      0.1084 |            0.9288 | flutter/flutter       | 2026-04-05 11:36:49+00:00 | 2026-04-05 11:52:49+00:00 |
| random_forest |      112322 |       2350 |            0.0536 |      0.1198 |   0.3889 | 0.1832 | 0.7748 |      0.1740 |            0.9380 | tensorflow/tensorflow | 2026-04-05 11:36:49+00:00 | 2026-04-05 15:50:44+00:00 |
| random_forest |      112769 |       2145 |            0.0499 |      0.0383 |   0.1028 | 0.0558 | 0.6605 |      0.1338 |            0.9207 | curl/curl             | 2026-04-05 11:36:49+00:00 | 2026-04-05 12:48:31+00:00 |

Uneven results are evidence against a single universal blocking policy; future work should use repository-specific data and calibration.
