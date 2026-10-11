# Temporal versus Random Split Evaluation

Both evaluations use the same strict pre-build `FEATURE_COLS` and frozen Random Forest hyperparameters. The random reference reproduces the production capped sampling methodology; the temporal evaluation uses all chronologically ordered builds, fitting encoders and imputation only on the earlier training period.

| Split | Train rows | Test rows | Train date range | Test date range | Precision | Recall | F1 | AUC | Training seconds |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Random (production methodology) | 160,000 | 40,000 | Interleaved 2011-2016 to Interleaved 2011-2016 | Interleaved 2011-2016 to Interleaved 2011-2016 | 0.3286 | 0.6217 | 0.4300 | 0.6902 | 1.75 |
| Temporal (earlier to later) | 2,849,851 | 712,463 | 2011-04-16 11:24:39+00:00 to 2016-03-07 14:53:32+00:00 | 2016-03-07 14:53:32+00:00 to 2016-08-31 23:59:27+00:00 | 0.2441 | 0.5177 | 0.3318 | 0.5910 | 134.32 |

On this dataset and timeframe, temporal F1 is 0.0982 points lower than the random-split reference (0.3318 versus 0.4300); temporal AUC differs by -0.0992. This is evidence only for this TravisTorrent export and its 2011-2016 chronology; it does not establish deployment readiness or generalization to other repositories/timeframes.
