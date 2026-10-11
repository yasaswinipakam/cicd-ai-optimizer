# Chronological Probability Calibration

Random Forest was fitted on the earliest model-training segment only. Sigmoid (Platt) and isotonic calibrators were fitted only on the subsequent chronological validation segment. The final temporal 20% was untouched until this evaluation.

Model-training rows: 2,564,865; calibration rows: 284,986; final-test rows: 712,463. Ranges: train through 2016-01-04 16:08:17+00:00, calibration through 2016-03-07 14:53:32+00:00, test from 2016-03-07 14:53:32+00:00.

| Method | Test Brier | Test ECE | Test AUC | Validation-selected 15% cutoff | Test alert rate | Tier precision | Tier failure capture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.1787 | 0.1691 | 0.6331 | 0.4602 | 15.56% | 0.3559 | 0.2825 |
| sigmoid | 0.1496 | 0.0291 | 0.6331 | 0.2879 | 15.56% | 0.3559 | 0.2825 |
| isotonic | 0.1491 | 0.0198 | 0.6271 | 0.2325 | 22.37% | 0.3116 | 0.3556 |

`sigmoid` was activated because it improved test calibration and preserved or improved the validation-selected 15% tier on the untouched test period. Calibration improves probability reliability; it is not evidence that the system prevents failures. The resulting policy remains risk prioritization and developer assistance only, never automatic CI blocking.
