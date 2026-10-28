# Next-stage plan: evidence, communication, and a financial-repository pilot

## Purpose

The project will make a narrow, testable claim: a leakage-safe risk ranking can help a team focus a fixed review budget on higher-risk changes. It will not claim universal failure prediction, guaranteed time savings, or superiority across every repository.

Three tracks are deliberately separated. A presentation artifact must not influence model selection; a financial pilot must not be mixed into the Travis/GitHub Actions estimates.

## Track 1 — close Phase 1 evidence

### Fixed headline

- Chronological, build-level Random Forest evaluation.
- Tier C remains the current candidate: strict pre-build fields, documented repository snapshots, and finish-time-safe local history.
- Macro within-repository AUC and per-repository top-15% recall/lift are primary. Pooled metrics are secondary and explicitly labelled as susceptible to repository-level separation.

### Remaining gates

1. Run Tier C walk-forward sensitivity analyses with one-hour and six-hour finish buffers; compare with zero-hour using paired repository-cluster bootstrap differences.
2. Complete corrected fixed Tier D variants (`DnormV2`, `DallV2`) and report every pre-specified result. Do not select variants on final-test performance.
3. Resolve the label-shuffle diagnostic: add independent random-score controls on the same saved test predictions and execute the pre-specified RF leaf/tree/sparse-score diagnostic. Report the below-0.5 shuffled null rather than treating it as a conventional p-value until explained.
4. Run the pre-specified per-project online protocol and report macro AUC and failure-class F1 beside prevalence.
5. Finish the GitHub Actions first-attempt primary evaluation, all-attempt sensitivity, eligible-repository holdouts, and a `failure` versus `startup_failure` outcome-stratified report.
6. Rerun the three Travis LORO units with the fully specified finish/start/build-ID tie-order protocol; retain prior parts as superseded artifacts.
7. Merge versioned parts into final CSVs; create `audit.md`, changed-number table, open questions, and a concise results summary.

### Stop rule

After the above fixed checks, freeze the Phase 1 feature set and results. Do not add features or alter windows based on scores.

## Track 2 — presentation-only demo

This track begins after Track 1 reporting is frozen. It uses a frozen model artifact and clearly labels all outputs as risk rankings, not diagnoses.

### Plain-language story

> Before an automated software safety check runs, the system ranks changes by estimated risk so a team can inspect the riskiest fraction first.

Explain the alert budget with a capture curve: if a team examines 15% of changes at random, it expects to find about 15% of failures. Report the observed per-repository capture rate with its uncertainty, never a pooled lift as an individual-team promise.

### Screens and claim boundaries

| Artifact | Allowed content | Not allowed until separately evaluated |
|---|---|---|
| Ranked risk board | Top-budget flag, calibrated risk score, timestamp, repository, plain-language feature summary | Claims that an individual change will fail |
| Capture curve | Saved chronological test predictions, random diagonal, repository-specific uncertainty | Pooled curve presented as one repository's result |
| Why-flagged panel | Existing pre-build signals, e.g. previous completed outcome, last-five failure rate, unusual change magnitude | File-level/payment-code explanations without path data |
| Leakage-audit slide | Build vs job distinction, random-vs-chronological split contrast, protocol safeguards | Unverified literature numbers or causal claims |

Failure diagnosis cards are out of scope for this research claim and must be a separate evaluated component.

## Track 3 — financial-repository pilot (separate Phase 2)

### Question

Does the frozen, leakage-safe ranking protocol show useful within-repository prioritization on a small, pre-declared collection of open-source financial-software repositories?

### Before any collection or modelling

1. Select a fixed set of repositories and record selection criteria, retrieval date, commit range, CI platform, and exclusions.
2. Define the prediction unit, failure label, first-attempt policy, retry policy, and pre-build time boundary.
3. Define criticality labels from source-controlled paths or configuration before looking at outcomes; distinguish a descriptive criticality badge from a predictive feature.
4. Write the chronological split, per-project online protocol, alert budget, primary metric, and minimum sample/failure eligibility rule.
5. State in advance that a null result is reportable and that the pilot will not change the Phase 1 headline model.

### Candidate contribution

The finance-specific contribution is a descriptive safety/criticality overlay for source-controlled areas such as payment, ledger, or authentication paths. It must be validated against a reproducible repository taxonomy and never be inferred from a post-failure diagnosis.

### Reporting

- Label it a pilot, including repository count and eligibility exclusions.
- Report every repository's prevalence, failures, AUC, recall@15%, precision@15%, and uncertainty.
- Keep finance results in separate tables and never pool them with TravisTorrent or GHA datasets.

## Communication rules

- Prefer: “risk ranking,” “alert budget,” “captured failures,” and “within-repository evaluation.”
- Avoid: “accurate prediction,” “prevents failures,” “saves money,” or “works for all teams.”
- State uncertainty and low-prevalence limitations next to repository-specific findings.
