# Optimization Impact Simulator - Design

## 1. Objective and Implementation Status

The proposed Optimization Impact Simulator will compare the production model's
prediction for an observed build input with its prediction for one carefully
bounded hypothetical input. It is a **model-based simulation**, not a causal
claim or a promise that a real build will pass.

Its central demo statement must be framed as:

> Under the current model, changing this hypothetical input changes predicted
> failure risk from X% to Y%.

It must never state that the change will reduce the actual failure rate by the
same amount.

**Implemented status:** the design below is now implemented at
`POST /predict/whatif/optimization` and in
`dashboard/src/pages/WhatIfPage.jsx`. The implemented response uses
`risk_delta = simulated_probability - baseline_probability` and the exact
disclaimer “Model-based simulation only. It does not guarantee the actual build
outcome or failure rate.”

## 2. Existing What-If Architecture

The existing sandbox is intentionally database-independent:

| Concern | Current implementation |
| --- | --- |
| Endpoint | `POST /predict/whatif` in `api/routers/predict.py` |
| Request | Existing `BuildFeaturesIn`; it accepts flat fields or a `build_features` object |
| Response | `failure_probability`, `is_anomaly` |
| Predictor path | `CICDPredictor.predict_failure()` and `detect_anomaly()` |
| Persistence | None; it creates neither `BuildLog` nor `Prediction` rows |
| Frontend | `dashboard/src/pages/WhatIfPage.jsx`, debounced by about 400 ms |
| Existing controls | source churn, test churn, canonical added/deleted/modified file counts, language, PR flag, previous-build-failed flag |

The predictor derives the production feature matrix in this order:

```text
prev_status_enc, lang_enc, is_master, by_core, is_pr, test_to_src_ratio,
total_files_changed, total_test_changes, hour, day_of_week, month,
is_weekend, is_night
```

It calculates `total_files_changed` from the canonical TravisTorrent fields:

```text
gh_diff_files_added + gh_diff_files_deleted + gh_diff_files_modified
```

`gh_team_size` is not a production model feature and is not exposed as a
What-If or optimization intervention. The current UI sends the three canonical
file-count fields so the derived total remains consistent with training.

## 3. Existing SHAP Architecture

For tree models, `CICDPredictor.full_analysis(..., include_explanation=True)`
uses `PredictionExplainer` and returns up to eight entries in this stable,
JSON-safe shape:

```json
{
  "feature": "total_files_changed",
  "value": 177.0,
  "shap_value": 0.060341397387089095,
  "direction": "increases_failure_risk"
}
```

SHAP is a local explanation of the current model prediction. It can rank a
feature as influential for a particular prediction; it does not prove that
changing that feature causes a real build result to change.

The root-cause suggestion engine is separate. It maps **post-failure** log
terms to troubleshooting advice. Its recommendations can provide context in
the UI but must not be translated automatically into model-feature changes.

## 4. Actionable Feature Candidates

Only raw inputs that deterministically affect current `FEATURE_COLS` should be
accepted. Derived model columns must never be set directly.

| Raw input / resulting feature | Candidate intervention | Valid bounds or values | Actionable? | Simulatable by current model? | Notes |
| --- | --- | --- | --- | --- | --- |
| `git_diff_src_churn` / `test_to_src_ratio` | Split a proposed code change into a smaller buildable unit | Integer >= 0; cap at the training-data supported UI range or an explicitly documented percentile bound | Yes, as planning scenario | Yes | Changing source churn also changes the derived ratio when test churn is fixed. |
| `git_diff_test_churn` / `total_test_changes`, `test_to_src_ratio` | Stage or split test-code changes into a smaller reviewable change | Integer >= 0; bounded as above | Yes, as planning scenario | Yes | Do not call this “reducing tests”; it describes hypothetical test-code churn. |
| `gh_diff_files_added`, `gh_diff_files_deleted`, `gh_diff_files_modified` / `total_files_changed` | Split a wide change into smaller independently buildable commits/PRs | Each integer >= 0; each requested field explicit | Yes | Yes | The API must preserve the canonical three-field formula. A “set total” request would be ambiguous and must be rejected. |
| `gh_is_pr` / `is_pr` | Compare the same hypothetical change as PR versus non-PR workflow context | Boolean only | Conditional | Yes | This is a workflow scenario, not a recommendation to bypass review. It should be hidden from default “recommended interventions.” |
| `git_branch` / `is_master` | Compare target-branch context | Allowed explicit branch labels; resulting feature is binary master/non-master | Conditional | Yes | Do not recommend merging directly to master; expose only as an advanced context scenario if needed. |
| `git_prev_commit_resolution_status` / `prev_status_enc` | None | Fitted encoder categories only | No | Technically yes | This is history, not an editable intervention. It can be shown as observed context only. |
| `gh_lang` / `lang_enc` | None | Fitted encoder categories only | No | Technically yes | Changing a repository language is not a build-level intervention. |
| `gh_by_core_team_member` / `by_core` | None | Boolean-like 0/1 | No | Technically yes | Identity/governance context, not a legitimate risk-control action. |
| Timestamp/time features | None | Derived from timestamp | No | Technically yes | Scheduling may be studied later, but the current model should not claim a causal timing intervention. |

The first implementation should expose only the first three rows by default:
source churn, test churn, and the canonical file-count components.

## 5. Proposed Intervention Model

Use **one intervention at a time** in version 1.

Why:

- It keeps the simulation interpretable: one changed raw input has one visible
  before/after value.
- It avoids arbitrary feature-search behavior that could generate implausible,
  adversarial combinations merely to minimize model probability.
- It avoids accidental conflicts between derived features, particularly
  `test_to_src_ratio` and the three-component file total.
- It makes viva language honest and easy to explain.

An intervention is a validated replacement for one permitted raw input or one
permitted file-count component. The system copies the full baseline raw build
dictionary, applies exactly one replacement, and scores the copy. It does not
mutate the request object, a `BuildLog`, a `Prediction`, or any artifact.

## 6. Simulation Algorithm

1. Receive a raw build-feature dictionary and one requested intervention.
2. Validate the intervention name against the explicit allow-list.
3. Validate its type and bounds. Reject unknown fields, derived
   `FEATURE_COLS` names, categorical free text, negative counts, and
   post-build `tr_log_*` fields.
4. Create a deep copy of the raw input.
5. Calculate the baseline through `CICDPredictor.predict_failure()`.
6. Obtain an optional baseline local SHAP explanation using the existing
   predictor/explainer path; never refit anything.
7. Apply the one validated replacement to the copy.
8. Calculate the simulated prediction with the same `CICDPredictor` and its
   saved imputer, encoders, feature order, and selected production model.
9. Compute:

   ```text
   risk_delta = simulated_failure_probability - baseline_failure_probability
   percentage_point_change = risk_delta * 100
   ```

10. Return both inputs' relevant derived values (particularly the canonical
    `total_files_changed`) and an explicit non-causal disclaimer.

There is no optimizer loop, no random search, no gradient search, and no
automated promise of the “best” action.

## 7. Proposed API Contract

Proposed endpoint: `POST /predict/whatif/optimization`.

It must require the same authentication as `/predict/whatif`, have no pipeline
identifier, and make no database write.

### Request

```json
{
  "build_features": {
    "git_diff_src_churn": 400,
    "git_diff_test_churn": 30,
    "gh_diff_files_added": 4,
    "gh_diff_files_deleted": 1,
    "gh_diff_files_modified": 18,
    "gh_lang": "ruby",
    "gh_is_pr": true
  },
  "intervention": {
    "field": "gh_diff_files_modified",
    "value": 8,
    "reason": "Split the proposed change into smaller independently buildable units"
  }
}
```

### Response

```json
{
  "simulation_type": "model_based_hypothetical",
  "baseline": {
    "failure_probability": 0.68,
    "predicted_status": "failed",
    "derived_features": {"total_files_changed": 23},
    "explanation": []
  },
  "simulation": {
    "failure_probability": 0.41,
    "predicted_status": "passed",
    "derived_features": {"total_files_changed": 13}
  },
  "risk_delta": -0.27,
  "percentage_point_change": 27.0,
  "intervention": {
    "field": "gh_diff_files_modified",
    "original_value": 18,
    "simulated_value": 8,
    "reason": "Split the proposed change into smaller independently buildable units"
  },
  "disclaimer": "Model-based simulation only. It does not guarantee the actual build outcome or failure rate."
}
```

`baseline.explanation` should be present only when existing SHAP support is
available; it must preserve the current explanation schema. The response must
not claim a causal effect.

## 8. Frontend UX

Extend the existing What-If page rather than redesigning the dashboard.

1. Keep the current **Live baseline failure probability** card.
2. Add a compact **Why this prediction** list using the existing local SHAP
   entries, when available.
3. Add a **Try one change** selector with only the allow-listed action inputs.
4. Show original value, bounded replacement control, and a plain-language
   planning label such as “Hypothetically split the change.”
5. On explicit “Simulate optimization” click, display:

   ```text
   Observed input -> Predicted current risk -> Simulated input
   -> Predicted simulated risk -> Estimated risk difference
   ```

6. Place a persistent disclaimer directly below the comparison:
   “Model-based simulation only. It does not guarantee the actual build outcome.”

Do not automatically apply a root-cause suggestion as a feature intervention.
When a root-cause suggestion exists, show it as separate post-failure advice.

## 9. Validation and Test Strategy

| Test | Expected invariant |
| --- | --- |
| Identical input / no-op rejected or same-value intervention | Baseline and simulated probability are identical; no random variation. |
| Valid source-churn intervention | Both predictions are valid floats in [0, 1]. |
| Valid canonical file-count intervention | Derived simulated `total_files_changed` equals added + deleted + modified. |
| Negative, nonnumeric, or out-of-range count | 422-style validation error; no prediction/write. |
| Unknown or derived feature name | Rejected; callers cannot set `total_files_changed`, `lang_enc`, or any `tr_log_*` field directly. |
| Categorical input | Arbitrary language/previous-status values rejected for intervention, even though inference itself has a safe unknown-category fallback. |
| Database integrity | `BuildLog` and `Prediction` counts unchanged before/after request. |
| Production path | Uses injected/current `CICDPredictor`; no new estimator, encoder, imputer, or artifact write. |
| Determinism | Same raw input and intervention produce the same baseline/simulated result. |
| SHAP compatibility | If explanation is returned, all entries retain existing JSON-safe keys/types. |
| Causal language | Response and UI assertions contain “model-based” and “not a guarantee”; avoid “will reduce.” |

## 10. Data Integrity Guarantees

- No `BuildLog`, `Prediction`, pipeline, dashboard aggregate, or SQLite record
  is created or changed.
- No classifier, imputer, scaler, encoder, anomaly detector, root-cause
  clusterer, or model-selection artifact is written.
- `FEATURE_COLS` stays the single source of truth and is selected/reordered only
  inside the existing predictor transformation.
- The simulator receives no `tr_log_*` field and never uses failed-test log
  text as a prediction input.
- Identical requests are deterministic because the frozen artifacts and
  deterministic feature transformation are reused.

## 11. Limitations and Viva Honesty

The simulator can say:

> Under the current frozen Random Forest model, hypothetically reducing changed
> files from 23 to 13 changes predicted failure risk from 68% to 41%.

It cannot say:

> Reducing changed files will reduce the actual build failure rate by 27%.

Why not:

- The model learned associations from historical TravisTorrent builds, not
  causal treatment effects.
- Change size can correlate with unobserved complexity, repository practices,
  and developer workflow.
- A hypothetical value may be outside the operational context of a real build.
- The model's F1 (0.4300) represents imperfect predictive performance.

The UI should label baseline information as **Observed**, the current score as
**Predicted**, and the altered scenario as **Simulated**.

## 12. Viva Demonstration Script

1. Open What-If with a sample proposed build and show the current predicted
   risk.
2. Show the local SHAP explanation for the current prediction; explain that it
   describes model contribution, not causality.
3. Select one action, for example reduce `gh_diff_files_modified` after
   splitting a wide change.
4. Click **Simulate optimization** and show baseline versus simulated risk plus the
   changed canonical file total.
5. Read the disclaimer: the number is a model-based scenario, not a promised
   operational result.
6. Confirm dashboard totals/history have not changed, proving that this was
   database-independent exploration.

## 13. Implementation Plan

1. Define a narrow intervention allow-list and Pydantic request/response models
   in the existing API schema module.
2. Add a database-free optimization simulation method that reuses
   `CICDPredictor`, without changing predictor model logic.
3. Add the authenticated `/predict/whatif/optimization` endpoint.
4. Add API tests for validation, deterministic results, canonical file totals,
   and unchanged database rows.
5. Add the compact selector/comparison/disclaimer to `WhatIfPage` using the
   existing API client.
6. Run targeted tests and the full suite; verify that no artifact timestamps
   change.

Implementation and validation are complete. The endpoint is covered by
`tests/test_optimization_simulator.py`; the dashboard uses the backend response
directly and does not calculate a local prediction.
