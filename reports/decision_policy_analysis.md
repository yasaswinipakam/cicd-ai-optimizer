# Decision Threshold Policy Analysis

This sweep uses only probabilities from the frozen production Random Forest on its existing deterministic 40,000-row test split. No estimator, calibrator, preprocessing object, artifact, or API threshold was changed. The R6 isotonic calibrator is not used here because it was research-only and not persisted; recreating it would require retraining.

| Policy | Threshold | Precision | Recall | Alert rate | Failure capture rate | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced | 0.50 | 0.3286 | 0.6217 | 0.4004 | 0.6217 | 0.4300 |
| Catch failures | 0.40 | 0.2585 | 0.8519 | 0.6973 | 0.8519 | 0.3967 |
| Avoid false alarms | 0.70 | 0.7488 | 0.0567 | 0.0160 | 0.0567 | 0.1054 |

For a developer warning before merge, **Balanced** (threshold 0.50) is the most reasonable starting recommendation because it maximizes F1 on this held-out split. If missed failures are more costly, **Catch failures** raises capture to 85.2% but alerts on 69.7% of builds. If alert fatigue is the dominant concern, **Avoid false alarms** raises precision to 74.9% while reducing captured failures to 5.7%. These are product trade-offs, not objectively correct settings; a product owner should set the operating policy using the cost of missed failures and false alerts.
