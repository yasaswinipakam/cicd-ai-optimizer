# Probability Calibration Analysis

The final 40,000-row production test partition was not used to fit either calibration method. Platt (sigmoid) and isotonic calibration were fitted through five-fold stratified calibration within the original training partition; each fold fits imputation and the fixed Random Forest configuration only on its fitting fold.

| Method | Brier Score | ECE | F1 | AUC |
| --- | ---: | ---: | ---: | ---: |
| Raw RF | 0.2154 | 0.2469 | 0.4300 | 0.6902 |
| Platt | 0.1524 | 0.0206 | 0.0887 | 0.6890 |
| Isotonic | 0.1512 | 0.0065 | 0.1577 | 0.6897 |

By Brier score, **Isotonic** is the most reliable method in this experiment (0.1512; ECE=0.0065). Compared with raw RF, its F1 changes by -0.2723 and AUC by -0.0005. Calibration changes the probability mapping rather than the underlying feature ranking, so little change in AUC is expected. This is a research recommendation only; no calibrated estimator has been deployed to the API.
