# Phase 2 finance pilot: results

**Current protocol:** v1.2 (amendment in `phase2_finance_pilot_protocol.md`)  
**Evaluated:** 2026-10-28  
**Unit:** commit/check-suite (first-attempt push/pull_request, SHA-collapsed)

## v1.2 corrected results (current headline)

### What changed and why

The v1.1 labels counted pull-request workflow runs that **never executed a
job** as failures. These were runs awaiting maintainer approval that GitHub
later expired (exactly 720 h) or failed when the PR closed; their `updated_at`
is that expiry. Job lists were fetched for all 2,821 Fineract first-attempt
runs labelled failure (`data/raw/phase2_finance_v1_jobs/`):

| | v1.1 | v1.2 |
|---|---:|---:|
| Failed runs that executed zero jobs | counted as failures | **1,588 of 2,821 (56%) dropped** |
| Commit/check-suites | 1,787 | 1,457 (330 had only never-run failures) |
| Failures / prevalence (all) | 1,002 / 0.561 | 712 / 0.489 |
| Test-window prevalence (final 20%) | 0.631 | **0.411** (120 of 292) |
| Suite duration, 90th percentile | 720 h | **1.6 h** |
| Max possible lift at 15% | 1.58× | 2.43× |

Zero-job failures appear across every PR-gated workflow (PR Compliance 758,
Fineract CI 269, Signed Commits Check 258, …), consistent with whole PRs never
being approved. The apparent August spike in failures (71% under v1.1) was
mostly expiries; corrected August prevalence is 39%. Fineract remains eligible
under unchanged thresholds (span 253 days, 120 final-20% failures). Audit:
`reports/phase2_finance_v1_2_suite_audit.json`. The other three candidates were
not re-checked for zero-job failures; they remain ineligible on volume/span.

Stream history (secondary variant) was verified against an independent
brute-force recomputation on 600 sampled suites at the 0 h and 6 h buffers.
Results were produced in the pinned project environment (Python 3.11,
scikit-learn 1.9.1); parts are in `reports/phase2_finance_v1_2_parts/`.

### Final development test (n=292, prevalence 0.411; weekly-block 95% CIs)

| Buffer | Model | AUC | 95% CI | Lift@15% | 95% CI | Recall@15% |
|---|---|---:|---|---:|---|---:|
| 0 h | C_rf (primary) | 0.549 | [0.475, 0.612] | 1.38 | [1.04, 1.71] | 0.208 |
| 0 h | Cstream_rf | 0.409 | [0.338, 0.484] | 0.94 | [0.60, 1.24] | 0.142 |
| 1 h | C_rf (primary) | 0.447 | [0.386, 0.514] | 1.05 | [0.75, 1.44] | 0.158 |
| 1 h | Cstream_rf | 0.527 | [0.459, 0.593] | 1.27 | [0.95, 1.49] | 0.192 |
| 6 h | C_rf (primary) | 0.534 | [0.474, 0.595] | 1.38 | [1.05, 1.62] | 0.208 |
| 6 h | Cstream_rf | 0.487 | [0.399, 0.570] | 1.00 | [0.63, 1.30] | 0.150 |

Paired Cstream − C AUC differences on the final test (weekly blocks, 10 weeks):
0 h −0.140 [−0.272, +0.006]; 1 h +0.080 [−0.052, +0.199]; 6 h −0.047
[−0.194, +0.092]. All intervals include zero and the sign flips across buffers.

### Online folds 6–10 (macro over folds)

| Buffer | C_rf AUC | Cstream_rf AUC | C_rf lift@15% | Cstream_rf lift@15% |
|---|---:|---:|---:|---:|
| 0 h | 0.523 | 0.586 | 1.11 | 1.24 |
| 1 h | 0.533 | 0.566 | 0.96 | 1.12 |
| 6 h | 0.551 | 0.585 | 0.93 | 1.14 |

Fold prevalence: 0.54, 0.27, 0.25, 0.42, 0.40. Stream history is ahead in
every buffer's fold macro, driven by folds 9–10 (AUC 0.62–0.69).

### Interpretation

- **The v1.1 "below chance" result was largely a labelling artifact.** With
  never-run checks removed, the primary model sits around chance to weakly
  positive (final AUC 0.45–0.55 across buffers, online macro 0.52–0.55).
- **There is no robust signal yet.** The primary model's final AUC moves from
  0.549 to 0.447 to 0.534 between buffers; lift@15% is above 1 at 0 h and 6 h
  but not at 1 h. With 292 test changes over 10 weeks, these swings are within
  sampling noise.
- **Stream (same-branch) history is the most promising lead, not a finding.**
  It is consistently ahead in the online folds (+0.03 to +0.06 macro AUC), but
  not on the final test, where paired differences change sign. It was
  pre-specified in v1.2, so it may be carried into a future protocol, but it
  should not be described as an improvement.
- Prevalence (41%) no longer constrains the analysis: the best possible lift is
  2.43×.

Everything below this line is the **superseded v1.1** analysis and the post-hoc
v2.0 exploration, kept for the record. Their prevalence (63%), AUCs and the
"null result" conclusion were computed on the contaminated labels.

---

## Cohort eligibility (v1.1)

| Repository | Commit suites | Failures | Prevalence | Span (days) | Final-20% failures | Eligible |
|---|---|---|---|---|---|---|
| apache/fineract | 1,787 | 1,002 | 0.561 | 253 | 226 of 358 | **Yes** |
| actualbudget/actual | 2,186 | 421 | 0.193 | 90 | 99 of 438 | No — span < 180 days |
| killbill/killbill | 210 | 76 | 0.362 | 390 | 10 of 42 | No — < 500 suites |
| firefly-iii/firefly-iii | 66 | 12 | 0.182 | 142 | 12 of 14 | No — < 500 suites |

One of four candidates is eligible. The three ineligible repositories are retained in the exclusion table; they are not modelled.

## Apache Fineract: development-test results

Primary model: 200-tree Random Forest, Tier C features (commit size + calendar + local history).  
Split: chronological final 20% as development test (n=358, failures=226, prevalence=0.631).  
Calibration selected by Brier score on validation window: sigmoid (Brier 0.2404 vs raw 0.2518 at 0 h buffer).  
Bootstrap confidence intervals: weekly-block, 1,000 resamples.

### ROC-AUC and ranking metrics

| Finish buffer | AUC | 95% CI | Recall@15% | 95% CI | Lift@15% | 95% CI |
|---|---|---|---|---|---|---|
| 0 h | 0.469 | [0.392, 0.550] | 0.146 | [0.119, 0.184] | 0.968 | [0.789, 1.215] |
| 1 h | 0.478 | [0.407, 0.538] | 0.133 | [0.096, 0.184] | 0.880 | [0.630, 1.212] |
| 6 h | 0.488 | [0.404, 0.565] | 0.150 | [0.118, 0.200] | 0.997 | [0.785, 1.319] |

**Finding:** All three buffer variants produce AUC point estimates below 0.5 with confidence intervals that span the random baseline. Lift at 15% is indistinguishable from 1.0 (no lift). This is a null result: the feature set does not provide useful prioritization on Apache Fineract in this observation window.

### Online folds (0 h buffer, folds 6–10)

| Fold | n | Failures | Prevalence | AUC | Lift@15% |
|---|---|---|---|---|---|
| 6 | 179 | 50 | 0.279 | 0.512 | 0.663 |
| 7 | 179 | 88 | 0.492 | 0.378 | 0.377 |
| 8 | 178 | 125 | 0.702 | 0.516 | 0.949 |
| 9 | 178 | 128 | 0.719 | 0.517 | 1.082 |
| 10 | 178 | 96 | 0.539 | 0.519 | 1.099 |
| **Macro** | | | | **0.489** | **0.834** |

Online macro AUC is 0.489. Folds 6–7 (low prevalence, earlier period) show the weakest signal. Folds 8–10 trend toward AUC ≈ 0.52 and lift ≈ 1.1, but these estimates have wide uncertainty at this sample size.

## Interpretation

### What limits the result: missing signal, not prevalence

Apache Fineract's failure prevalence in the test window is 63%: almost two in three commit/check-suites triggered at least one failed workflow run. High prevalence does cap the *lift* any ranking can achieve: a perfect model flagging the top 15% would reach at most 1/0.631 = **1.58×**. But the model reached **0.97×**, far below that ceiling, so the ceiling is not what binds.

ROC-AUC does not depend on prevalence at all. An AUC of 0.469 (95% CI 0.392–0.550) means the features do not rank failing changes above passing ones in this window. The binding constraint is missing signal in the Tier C feature set on this repository, not the failure rate. The online folds agree: fold 6, at only 28% prevalence, is also at chance (AUC 0.512).

### What this result does and does not say

**Does say:** The Tier C Random Forest feature set (commit size, calendar signals, prior completed-run history) does not produce reliable above-random prioritization on Apache Fineract in the Feb–Oct 2026 observation window, where failure rates were high and rising.

**Does not say:** That risk ranking is ineffective for finance software in general, or on other repositories, or in observation windows with lower failure prevalence. The three ineligible repositories could not be evaluated.

**Does not say:** That any of the other three cohort repositories would show the same result. Actualbudget/actual and Killbill/killbill each had lower observed prevalence (0.19 and 0.36 respectively) but insufficient data volume.

## Exploratory v2.0: LightGBM + finance criticality badges (post-hoc, not a protocol result)

**Protocol deviation.** Frozen protocol v1.1 declares the path badges a descriptive overlay only and forbids changing features or models after reading model metrics. v2.0 was run *after* the v1.1 null result, so it is a post-hoc exploration: it can generate hypotheses for a future pre-registered protocol, but it cannot revise the v1.1 finding.

v2.0 added five finance path-criticality badges as features and a LightGBM model (500 trees, learning_rate=0.05, scale_pos_weight) alongside the baseline RF. Badge coverage: 44.5% of suites touch at least one finance-critical path (payment/billing, ledger/accounting, auth/security, dep/ci). v2.0 was run in the Anaconda environment (Python 3.13, scikit-learn 1.7.2), not the project's pinned environment.

### v2.0 online fold comparison (0 h buffer)

| Fold | n | Prev | RF AUC | LightGBM AUC |
|---|---|---|---|---|
| 6 | 179 | 0.279 | 0.512 | 0.484 |
| 7 | 179 | 0.492 | 0.378 | 0.375 |
| 8 | 178 | 0.702 | 0.516 | 0.450 |
| 9 | 178 | 0.719 | 0.517 | 0.538 |
| 10 | 178 | 0.539 | 0.519 | 0.527 |
| **Macro** | | | **0.488** | **0.475** |

### v2.0 final dev-test (0 h buffer)

| Model | Features | AUC | Lift@15% |
|---|---|---|---|
| RF (Tier C) | Commit size + calendar + history | 0.469 | 0.968 |
| LightGBM (Tier E) | + finance criticality badges | 0.485 | 0.968 |

**Finding (v2.0, exploratory):** no improvement. The final dev-test AUC difference is +0.016 (0.469 → 0.485), with a paired weekly-block bootstrap 95% CI of **[−0.026, +0.062]** (1,000 resamples, 8 weeks), so it is indistinguishable from zero. The online fold macro AUC is lower for LightGBM (RF 0.488 vs LightGBM 0.475). Lift at 15% is identical (0.968) for both models, against a possible 1.58×. Both models remain at chance on this window.

## Comparison with Phase 1

| Dataset | Model | Within-repo AUC | Lift@15% | Repositories |
|---|---|---|---|---|
| Phase 1 (TravisTorrent, walk-forward folds 6–10) | Tier C RF | 0.631 (macro) | 1.85 (macro) | 511–544 per fold |
| Phase 1 (TravisTorrent, confirmation folds 7–10) | Tier E LightGBM rank/binary blend | 0.698 (macro) | 2.23 (macro) | 511–544 per fold |
| Phase 2 v1.1 (Fineract, protocol) | Tier C RF | 0.469 final / 0.489 online macro | 0.97 final / 0.83 online | 1 |
| Phase 2 v2.0 (Fineract, post-hoc) | LightGBM + badges | 0.485 final / 0.475 online macro | 0.97 final | 1 |

The two phases are not directly comparable: Phase 1 is a macro average over hundreds of repositories with prevalence-free ranking metrics, Phase 2 is a single repository. Phase-1 repositories average **15.2%** failure prevalence (pooled 16.7%, fold 8), lower than Fineract's 63%, but prevalence is not what separates the results. Phase 1 has real within-repository ranking signal (AUC 0.63–0.70); on Fineract the same feature family has none (AUC ≈ 0.47–0.49). Whether Actualbudget/actual (19% prevalence) would show signal once it becomes eligible is an open question, not an expectation.

## Limitations

- Single eligible repository: the pilot cannot support multi-repository averages or claims about finance software as a domain.
- Historically truncated extraction: both Fineract and Actualbudget reached GitHub's pagination ceiling; data represents the newest accessible window only.
- The protocol (v1.1) feature set is restricted to commit size, calendar, and local CI history; path-level criticality badges were a descriptive overlay in v1.1 and were used as predictors only in the post-hoc v2.0 exploration.
- With 358 test changes over 8 weeks, uncertainty is wide (AUC CI width ≈ 0.16), so small differences between models cannot be resolved.
- The three ineligible repositories may become eligible with additional data collection over a longer time window.

## Required artifacts checklist

- [x] `phase2_finance_collection_manifest.json` (from collection scripts)
- [x] `phase2_finance_data_audit.md`
- [x] `phase2_finance_commit_suite_eligibility.csv`
- [x] `phase2_finance_feature_audit.md`
- [x] `reports/phase2_finance_v1_parts/` (versioned predictions, all buffers × all folds)
- [x] `phase2_finance_results.md` (this file)

## Next steps

1. Extend data collection after six months to re-evaluate Actualbudget/actual (will exceed 180-day span).
2. Explore whether a lower-prevalence observation window on Fineract (2025 period) changes the result.
3. Consider expanding the finance cohort with additional eligible repositories before drawing domain-level conclusions.
