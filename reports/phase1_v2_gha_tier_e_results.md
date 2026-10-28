# External test of the Tier E recipe on GitHub Actions — results

The protocol is [`phase1_v2_gha_tier_e_protocol.md`](phase1_v2_gha_tier_e_protocol.md),
written before any model was fit. Its SHA-256 (`b4121c6a…4186`) is recorded in
`phase1_v2_gha_tier_e_manifest.json`. Code: `scripts/evaluate_gha_tier_e.py`.

**Data.** The test uses first-attempt `push` and `pull_request` runs from 30
repositories, covering 2025–2026: 67,980 runs, of which 8.9% failed. 687 runs
with an unknown completion time were dropped.

**Split.** Walk-forward over 10 chronological folds, with test folds 6–10. Each
test fold has about 6,800 runs, and 16–20 repositories have both outcomes.

## Pre-registered verdict: inconclusive (direction replicates)

| Within-repo AUC, mean of folds 6–10 (0h) | GHA | Travis (folds 7–10) |
|---|---:|---:|
| Tier C RF | 0.715 | 0.628 |
| **Tier E blend** (no job counts) | **0.770** | 0.695 |
| Δ E − C | **+0.055** | +0.067 |
| Fold CIs above 0 | 3 of 5 | 4 of 4 |

Per-fold Δ: +0.050 / +0.071 / +0.036 / +0.110 / +0.008.

- **The direction replicates.** Δ is positive in every fold.
- **The size is similar to Travis** (+0.055 vs +0.067).
- **The pre-registered bar was not met.** It required 4 of 5 fold CIs to
  exclude 0, and only 3 do. Fold 6 clears zero only barely, with a lower bound
  of +0.0004.
- **Why the CIs are wide:** with 16–20 repositories per fold, each fold's CI
  spans about ±0.06.

**Recall@15% replicates more clearly.** Tier E catches 48% of failures in a
15% within-repo review budget, against 37% for Tier C (lift 3.07× vs 2.33×).
The delta's CI excludes 0 in 4 of 5 folds. Pooled AUC is 0.870 vs 0.833.

## Secondary results

| Setting | Tier C RF | Tier E blend | Δ | Fold CIs > 0 |
|---|---:|---:|---:|---:|
| 0h, all runs (primary) | 0.715 | 0.770 | +0.055 | 3/5 |
| 0h, label-validity sensitivity | 0.703 | 0.754 | +0.051 | 3/5 |
| 0h, new breakages only | 0.645 | 0.695 | +0.050 | 0/5 |
| 6h buffer, all runs | 0.698 | 0.718 | +0.021 | 0/5 |
| 6h buffer, new breakages | 0.665 | 0.682 | +0.017 | 0/5 |

The label-validity sensitivity drops startup failures and zero-duration
failures.

### Differences from Travis

- **The 6 h buffer removes most of the gain.** On Travis the gain *grew* with
  stricter buffers.
  - Most GHA runs finish within an hour, so on GHA the useful signal is the
    very recent outcome of the same stream. A 6 h delay withholds it.
  - On Travis, the in-flight features carried the gain when outcomes were
    withheld.
- **Never-executed runs don't drive the result.** Dropping proxies for runs
  that never executed (648 runs) leaves the result almost unchanged. This
  data does not have the Fineract labelling artifact.
- **New breakages.** Tier E scores 0.695 against 0.645 on new breakages,
  where the previous run of the same workflow on the same stream passed. The
  persistence baseline scores 0.479. The gap is similar to all runs, but
  there are only about 127 such failures per fold, so no fold's CI excludes
  0.
- **Attribution differs** (reported, not selected).
  - The ranking objective alone adds little here: `C_lgbmblend` scores
    0.725, only +0.010 over Tier C RF.
  - The stream and in-flight features carry the gain: `E_lgbm` scores 0.770.
  - On Travis, the ranking objective was the largest single lever.

Other baselines, 0h: features without history (`A_rf`) 0.673; last-5
repository failure rate 0.540; repository prior 0.500.

## Disclosed implementation detail

GHA creation times have one-second resolution. The first run stopped because
an equal-size fold boundary split runs created in the same second, and the
train-before-test assertion failed. Each boundary is now moved forward past
tied creation times, keeping the protocol's strict train-before-test rule. No
results were produced by the failed run.

## Reading

- **The same direction and similar size on a different CI platform** support
  the claim that per-branch/PR history and in-flight context generalise.
- **But the gain is not established at the bar set in advance.** It is fragile
  to delaying outcomes by 6 h.
- **The right claim:** "replicates in direction and size; inconclusive under
  the pre-registered rule with 30 repositories". It is not a confirmed
  replication.
- **What would settle it:** more repositories, rather than more folds.
