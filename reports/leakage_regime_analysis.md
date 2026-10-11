# Leakage Regime Gradient Experiment

All rows use the same 200,000-row cap, deterministic 80/20 stratified split, random seed, and Random Forest hyperparameters. The strict regime is exactly the production `FEATURE_COLS`.

| Regime | Features | Precision | Recall | F1 | AUC | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict Pre-Build | 13 | 0.3286 | 0.6217 | 0.4300 | 0.6902 | 1.76 |
| Post-Build / Leaky | 31 | 0.6782 | 0.6973 | 0.6876 | 0.8956 | 2.87 |

For this dataset, adding post-build log/duration information changes F1 from 0.4300 to 0.6876: an absolute increase of 0.2577 points and a relative increase of 59.9%. This result is limited to this TravisTorrent export and this controlled feature/split configuration.

No distinct genuine during-build regime was evaluated: the registry identifies only `tr_jobs`, a mutable, ambiguously timed collection rather than a timestamped in-flight measurement. It was not added.

Post-build/leaky fields: `tr_log_lan`, `tr_log_status`, `tr_log_setup_time`, `tr_log_analyzer`, `tr_log_frameworks`, `tr_log_bool_tests_ran`, `tr_log_bool_tests_failed`, `tr_log_num_tests_ok`, `tr_log_num_tests_failed`, `tr_log_num_tests_run`, `tr_log_num_tests_skipped`, `tr_log_num_test_suites_run`, `tr_log_num_test_suites_ok`, `tr_log_num_test_suites_failed`, `tr_log_tests_failed`, `tr_log_testduration`, `tr_log_buildduration`, `tr_duration`.
