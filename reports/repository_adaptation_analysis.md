# Repository-Aware Adaptation Study

For each repository, a global Random Forest was fitted only on other repositories with timestamps before the target repository's final-test window. The target's earliest 10% of builds fitted an isotonic probability mapper, the next 10% selected a threshold from 0.05–0.95 by F1, and the later 80% remained untouched until final reporting. This is adaptation of calibration and policy, not a production model replacement or Random Forest fine-tuning.

| Repository | Method | Calibration / policy / final-test builds | Threshold | Alert rate | Precision | Recall | F1 | AUC | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DataDog/dd-agent | Global RF (default threshold) | 13,288 / 13,289 / 106,311 | 0.50 | 0.1501 | 0.5102 | 0.1397 | 0.2194 | 0.4243 | 0.3060 | 0.2021 |
| DataDog/dd-agent | Global RF + target isotonic calibration + target policy threshold | 13,288 / 13,289 / 106,311 | 0.05 | 1.0000 | 0.5478 | 1.0000 | 0.7079 | 0.4214 | 0.3152 | 0.1974 |
| pyca/cryptography | Global RF (default threshold) | 10,876 / 10,877 / 87,014 | 0.50 | 0.0756 | 0.0255 | 0.0632 | 0.0364 | 0.5197 | 0.1472 | 0.3303 |
| pyca/cryptography | Global RF + target isotonic calibration + target policy threshold | 10,876 / 10,877 / 87,014 | 0.05 | 0.2215 | 0.0199 | 0.1445 | 0.0350 | 0.5180 | 0.0309 | 0.0311 |
| opf/openproject | Global RF (default threshold) | 10,157 / 10,158 / 81,261 | 0.50 | 0.4897 | 0.3537 | 0.5384 | 0.4269 | 0.5549 | 0.2525 | 0.1637 |
| opf/openproject | Global RF + target isotonic calibration + target policy threshold | 10,157 / 10,158 / 81,261 | 0.15 | 0.9976 | 0.3215 | 0.9971 | 0.4862 | 0.5517 | 0.2183 | 0.0429 |

`DataDog/dd-agent`: F1 0.2194 → 0.7079; Brier 0.3060 → 0.3152. `pyca/cryptography`: F1 0.0364 → 0.0350; Brier 0.1472 → 0.0309. `opf/openproject`: F1 0.4269 → 0.4862; Brier 0.2525 → 0.2183. For `DataDog/dd-agent`, the F1 increase is driven by the selected 0.05 threshold alerting 100% of later builds; it is therefore not evidence of useful selective ranking. For `pyca/cryptography`, calibration greatly improves Brier/ECE but does not improve F1. Results are restricted to the three largest repositories, selected in advance by build count, and do not establish that one adaptation policy will transfer to every repository.
