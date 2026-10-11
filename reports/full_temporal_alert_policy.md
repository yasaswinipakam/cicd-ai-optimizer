# Full Temporal Alert-Volume Policy Analysis

This analysis uses only frozen full-mode Random Forest probabilities for the untouched later 20% chronological test period. It does not retrain, calibrate, or change the live threshold. Because raw probabilities have not yet been recalibrated for this full temporal model, score cutoffs should be treated as risk-ranking policies rather than literal failure likelihoods.

| Target alert rate | Score threshold | Achieved alert rate | Precision | Failure capture | F1 | Alerts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 0.5874 | 5.00% | 0.6327 | 0.1614 | 0.2572 | 35,625 |
| 10% | 0.5119 | 10.00% | 0.4596 | 0.2345 | 0.3105 | 71,247 |
| 15% | 0.4778 | 15.00% | 0.3911 | 0.2994 | 0.3391 | 106,871 |
| 20% | 0.4522 | 20.00% | 0.3620 | 0.3694 | 0.3657 | 142,496 |
| 35% | 0.4005 | 35.00% | 0.3078 | 0.5497 | 0.3946 | 249,363 |

For an initial shadow deployment, the 15% top-risk policy (score ≥ 0.4778) is a practical review starting point: it produces alerts for 15.0% of later builds and captures 29.9% of failures. It is a recommendation for measuring operational utility, not an automatic build-blocking rule.
