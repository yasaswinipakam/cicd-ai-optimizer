# Final End-to-End Consistency and Viva Readiness Audit

**Audit date:** 2026-08-31  
**Scope:** Read-only consistency audit of the frozen real-data model/application workflow. No model, artifact, database, dependency, or source-code changes were made.

> **Resolution update (Task 11):** The four actionable audit findings were
> subsequently resolved without retraining or changing artifacts: manual submit
> now sends canonical three-part file counts, empty failed-test logs return no
> root cause, anomaly wording describes an unusual pre-build input profile, and
> stale reports were synchronized. The findings below remain as the historical
> audit record that motivated that follow-up.

## 1. Executive Summary

The system is a coherent CI/CD build-risk platform with a real TravisTorrent-derived training dataset, a frozen pre-build Random Forest prediction model, authenticated FastAPI workflows, post-failure root-cause clustering, local SHAP attribution, PDF reports, and database-free What-If/optimization simulation.

The pre-build classifier is leak-free with respect to `tr_log_*` fields. The canonical production file-count feature is consistently defined as:

```text
gh_diff_files_added + gh_diff_files_deleted + gh_diff_files_modified
```

The frozen production model is **Random Forest**, selected by held-out F1. Its honest current held-out metrics are precision **0.3286**, recall **0.6217**, F1 **0.4300**, and AUC **0.6902**. The old notebook F1 of 0.755 must not be presented as production performance.

No blocker was found. At the time of this audit, the main findings were presentation/contract/documentation issues; they are recorded below as historical evidence and were resolved in the Task 11 follow-up described above.

## 2. Architecture Consistency

| Layer | Implementation | Audit result |
| --- | --- | --- |
| Data collection | `src/data/`, `api/routers/webhooks.py`, `scripts/simulate_ci_traffic.py` | Present; GitHub uses HMAC validation and Jenkins uses a shared secret. |
| Feature engineering | `src/features/engineer.py`, `src/features/schema.py` | Single canonical feature list with explicit post-build exclusion. |
| Prediction | `src/models/train.py`, `src/models/evaluate.py`, `src/inference/predictor.py` | Frozen artifacts and inference use the same 13-input schema. |
| Anomaly detection | `src/models/anomaly.py`, `src/models/train_anomaly.py` | Uses the same pre-build, imputed 13-feature representation as prediction. |
| Root-cause analysis | `src/models/clustering.py`, `src/models/train_clustering.py` | Separate post-failure TF-IDF/KMeans analysis of `tr_log_tests_failed`. |
| API/persistence | `api/` | Authentication, ingestion, persistence, PDF, history, What-If, and optimization endpoints present. |
| Dashboard | `dashboard/src/` | Production build succeeds; pages consume API endpoints through Axios bearer-token handling. |

## 3. Feature Consistency

`FEATURE_COLS` is the single source of truth in `src/features/schema.py`:

```text
prev_status_enc, lang_enc, is_master, by_core, is_pr, test_to_src_ratio,
total_files_changed, total_test_changes, hour, day_of_week, month,
is_weekend, is_night
```

Read-only artifact inspection confirmed:

- `FEATURE_COLS == artifacts/preprocessing.joblib["feature_cols"]`: **true**.
- Random Forest `n_features_in_`: **13**.
- Predictor `feature_cols == FEATURE_COLS`: **true**.
- Persisted encoders: `prev_status_enc`, `lang_enc`.

Offline training computes `total_files_changed` in `src/features/engineer.py::_total_files_changed`; live inference mirrors it in `src/inference/predictor.py::_total_files_changed`. Both prefer the canonical three `gh_diff_files_*` inputs and only retain a legacy `git_diff_src_files + git_diff_test_files` fallback for older clients.

Training fits/persists encoders only in `src/models/train.py`. Evaluation and anomaly training explicitly call `engineer_features(..., encoders=..., fit_encoders=False, save_encoders=False)`. Live inference calls only encoder `transform`; it does not fit or save encoders.

## 4. Leakage Status

**PASS for the pre-build classifier.** `tr_log_*` fields are absent from `FEATURE_COLS`, and `engineer_features()` explicitly documents their exclusion. The target `tr_status` is used only to construct the offline label.

Legitimate post-build data is kept separate:

- `tr_log_tests_failed` is used only by root-cause clustering after failure.
- `failed_test_log` is persisted as log content and is not selected by the classifier feature schema.
- Prediction explanations and recommendations are downstream outputs, not classifier inputs.

The optimization schema rejects `tr_log_*` inputs. The general build-ingestion schema can retain arbitrary raw fields for auditability, but the predictor selects only its fixed 13 engineered inputs; unrecognized raw fields do not enter the classifier.

## 5. Model and Metric Status

`artifacts/model_results.csv` and `reports/final_results.md` agree exactly:

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.6511 | 0.3286 | 0.6217 | **0.4300** | **0.6902** |
| Decision Tree | 0.5748 | 0.2729 | 0.6064 | 0.3764 | 0.6170 |
| SVM | 0.5618 | 0.2579 | 0.5701 | 0.3551 | 0.5855 |
| XGBoost | 0.6278 | 0.3066 | 0.6013 | 0.4061 | 0.6643 |

`artifacts/best_model.txt` is `random_forest`; `CICDPredictor` loads that exact artifact. These are deterministic metrics on the reproducible 20% held-out portion of the fixed 200,000-row sample, not training-set metrics.

`reports/regression_diagnosis.md` correctly rejects notebook F1 0.755 as non-deployable/leakage-affected evidence. Its TP-count prose was synchronized to the displayed confusion matrix (TP=5,263) in the Task 11 follow-up. The current F1/precision/recall table aligns with the stored CSV.

## 6. SHAP Status

**PASS.** `PredictionExplainer` is built with the currently loaded production tree model and the predictor's persisted feature order. It receives the already-imputed one-row feature frame, converts values to native Python floats, and returns local contributions sorted by absolute magnitude.

- API persistence: `create_build(...?explain=true)` stores the returned explanation under `raw_features["_explanation"]`.
- History: `list_builds()` returns that persisted explanation.
- PDF: `download_build_report()` passes the persisted explanation to `generate_build_report_pdf()`.
- Dashboard: `PipelineDetailPage` renders a SHAP contribution bar chart.
- Optimization: `POST /predict/whatif/optimization` can include the baseline explanation.

SHAP is used as local model attribution only. It is not causal evidence.

## 7. What-If Status

**PASS.** `POST /predict/whatif` requires authentication, invokes `CICDPredictor.predict_failure()` and `detect_anomaly()`, and does not create a `BuildLog`, `Prediction`, or pipeline. `tests/test_predict_whatif.py` verifies database row counts are unchanged.

The current What-If UI exposes raw features that affect the model and no longer presents `gh_team_size`. It sends all three canonical `gh_diff_files_*` inputs, so its derived file total is preserved.

## 8. Optimization Simulator Status

**PASS.** `POST /predict/whatif/optimization` is authenticated, database-free, deterministic, and uses the current `CICDPredictor` rather than a browser-side model.

- Exactly one intervention is accepted.
- Only five bounded raw action fields are allowed: source churn, test churn, files added, files deleted, and files modified.
- Derived features and `tr_log_*` fields are rejected.
- Input is deep-copied before changing the scenario.
- `risk_delta` is correctly implemented as `simulated_probability - baseline_probability`.
- The UI displays backend-returned baseline/simulated values and the exact non-causal disclaimer.

`tests/test_optimization_simulator.py` covers allowed fields, invalid fields/values, determinism, canonical totals, and no database mutation.

## 9. Anomaly Status

The detector is an **unusual pre-build input-profile detector**, not a failure classifier and not an execution-duration/profile detector. It is trained and scored on the same imputed 13-feature pre-build representation as the classifier.

Read-only reproduction against the frozen detector produced:

| Group | Builds | Failure rate |
| --- | ---: | ---: |
| Anomalous | 4,000 | 21.175% |
| Normal | 36,000 | 21.161% |

There is no meaningful observed failure-rate increase among anomalies. Therefore anomaly status must be described as an input-profile outlier signal that may justify investigation, not as evidence that a build is likely to fail.

## 10. Root-Cause Status

Root-cause clustering is correctly separated from prediction. It uses only failed rows' `tr_log_tests_failed` strings, with empty logs assigned to a non-test bucket. Its labels are TF-IDF/KMeans cluster themes, not demonstrated causes.

The current inference service calls `analyze_root_cause()` even when a request has no failed-test log, which yields the non-test bucket and persists a label. This is not classifier leakage, but it can make passed/manual builds appear to have a root cause. See Findings.

## 11. Suggestion Status

Suggestions are rule-based mappings from post-failure root-cause terms, with an optional anomaly note. Passed predictions receive no optimization suggestion in `CICDPredictor.full_analysis()`.

The text uses recommendations such as “consider” and “review”; it does not promise a fix. Suggestions should remain framed as troubleshooting guidance rather than causal proof.

## 12. API and Database Status

| Workflow | Result |
| --- | --- |
| JWT login/current user | Implemented and tested. |
| Role dependency | Implemented; no current routes are specifically role-restricted. |
| GitHub webhook | HMAC-SHA256 raw-body validation before JSON/database work; tested for valid, tampered, missing, and malformed signatures. |
| Jenkins webhook | Shared-header secret; documented provider asymmetry. |
| Build ingestion/persistence | Implemented and tested with in-memory SQLite. |
| Explanations/history/PDF | Implemented and tested, including persisted explanation retrieval. |
| What-If/optimization | Authenticated and verified database-free by tests. |

Nullable output handling is generally safe. Explanations use a consistent `_explanation` database-storage key and `explanation` response key. The storage/key difference is internal but should remain documented.

## 13. Frontend/Backend Contract Status

The following contracts match their schemas:

- What-If: nested `build_features`, returns failure probability and anomaly status.
- Optimization: nested `build_features`, one typed intervention, optional explanation; UI uses server-calculated delta rather than JavaScript prediction logic.
- Pipeline history: API returns persisted explanations, suggestions, root-cause labels and PDF endpoint; detail page consumes them.
- PDF: dashboard uses Axios blob download, retaining bearer authorization.

One manual-submit inconsistency remains: `SubmitBuildPage` labels a single input “Files changed” but sends `git_diff_src_files`, which activates the legacy fallback rather than the canonical three-component file-count mapping used offline and on the What-If page. See Findings.

## 14. Test Coverage Status

| Capability | Meaningful automated coverage |
| --- | --- |
| Preprocessing | `test_preprocess.py` |
| Feature engineering/canonical total/encoder transform-only behavior | `test_features.py` |
| Training artifact evaluation CSV bounds | `test_evaluate.py` |
| Predictor sparse/unseen categories | `test_predictor.py` |
| Anomaly detector | `test_anomaly.py` |
| Root-cause clustering | `test_clustering.py` |
| Suggestions | `test_suggestions.py` |
| SHAP JSON-safe output | `test_explainer.py` |
| PDF generation and explanation content | `test_pdf_report.py` |
| Authentication/database/API persistence | `test_auth.py`, `test_db.py`, `test_pipelines.py`, `test_e2e.py` |
| What-If no-write invariant | `test_predict_whatif.py` |
| Optimization no-write/validation/determinism | `test_optimization_simulator.py` |
| Webhooks/signature validation | `test_webhooks.py`, `test_webhook_signature.py` |
| Frontend | Production build and lint only; no frontend component/browser test runner is configured. |

Latest test run after Task 11: `.venv/bin/python -m pytest tests/ -v --tb=short` → **34 passed**. Frontend `npm run build` passed. `npm run lint` passed with one existing Fast Refresh warning in `dashboard/src/auth/AuthContext.jsx`.

## 15. Documentation Status

The README provides setup, model-artifact preparation, Docker startup, and GitHub signature guidance. `final_results.md` accurately reports the current metrics. `regression_diagnosis.md` correctly explains the rejection of 0.755, subject to the numerical prose discrepancy noted above.

The following documentation was synchronized in Task 11:

- `reports/traceability.md` now reflects real data/artifact availability and current endpoint coverage.
- `reports/demo_script.md` now includes the completed SHAP/optimization flow and accurately identifies Docker runtime—not artifacts—as the timing limitation.
- `reports/optimization_simulator_design.md` now records the implemented `risk_delta` contract and current What-If controls.

## 16. Security Status

Security controls verified statically:

- Passwords are bcrypt hashes.
- JWT uses HS256; production mode rejects an absent `JWT_SECRET`.
- GitHub webhook signature comparison uses `hmac.compare_digest` over raw request bytes.
- Optimization uses an explicit allow-list and bounds for actionable inputs.
- Simulations do not write database rows.

Operational caveats:

- Development has intentional fallback/default credentials and a development JWT secret; production must set environment secrets.
- CORS is restricted to `http://localhost:5173`, appropriate for the local demo but not a deployed frontend origin.
- Pipeline names are not database-unique and are resolved globally, so multi-user ownership isolation is not yet enforced.

## 17. Docker Status

**CONFIGURATION VERIFIED; RUNTIME NOT VERIFIED.** Docker was not run because Docker-daemon access was unavailable during the project validation.

- `requirements.txt` includes `shap` and `reportlab`.
- API image: Python 3.11 slim, installs requirements, copies `src`, `api`, `artifacts`, and `data`, initializes tables, then runs Uvicorn on port 8000.
- Dashboard image: Node 20 build stage runs `npm ci`/`npm run build`; Nginx serves port 80, mapped to host 5173.
- Compose maps API `8000:8000`, dashboard `5173:80`, mounts `./artifacts`, and uses a named SQLite volume at `/app/data`.
- Compose injects `JWT_SECRET`, `ADMIN_PASSWORD`, and `GITHUB_WEBHOOK_SECRET`; dashboard starts after API.

## 18. Recommended 5–7 Minute Viva Demonstration

1. Start the API/dashboard and log in with a configured account.
2. Show Dashboard summary cards and a pipeline history record.
3. Open a failed build: explain the predicted probability, then explicitly state that anomaly status is an unusual **pre-build input profile**, not a failure prediction.
4. Show the SHAP chart: it attributes this model prediction and is not causal evidence.
5. Show failed-build root-cause cluster terms and recommendation; state they are post-failure cluster themes and troubleshooting suggestions, not proven causes.
6. Download the per-build PDF and show the persisted explanation/suggestion.
7. Open What-If, alter a raw pre-build input, and point out it does not change dashboard totals.
8. Select exactly one approved optimization intervention, run it, and show baseline risk, simulated risk, canonical total-files-changed, and `risk_delta`.
9. Read the UI disclaimer: the simulation is model-based and does not guarantee a build result or actual failure-rate change.
10. Show `reports/final_results.md` and state the defensible Random Forest F1 is 0.4300, not 0.755, because post-build fields are excluded.

## 19. Historical Findings Classified by Severity (Resolved in Task 11)

### BLOCKER

None found.

### IMPORTANT

1. **Manual submit bypasses the canonical client mapping**  
   - File/component: `dashboard/src/pages/SubmitBuildPage.jsx`, `submit`.  
   - Issue: the “Files changed” field sends `git_diff_src_files`, invoking live predictor compatibility fallback rather than the canonical three `gh_diff_files_*` values used by offline processing and What-If.  
   - Why it matters: affects feature-consistency evidence and may make the demo imply that “source files” equals total changed files.  
   - Impact: correctness/demo quality. The fallback is intentional and does not corrupt the classifier, but the UI should use a canonical three-component input or be relabeled clearly.

2. **Root cause is attached without a failed-test log or confirmed failure**  
   - File/function: `src/inference/predictor.py`, `CICDPredictor.full_analysis`; persisted by `api/routers/pipelines.py`, `create_build`.  
   - Issue: `analyze_root_cause(failed_test_log)` runs for every analysis, including a missing log, producing the non-test bucket and a root-cause label.  
   - Why it matters: passed or predictive-only manual builds can display/store a root cause, whereas root-cause clustering is intended for already-known failures.  
   - Impact: demo quality/documentation correctness; no classifier leakage.

3. **Anomaly terminology overstates its semantics**  
   - Files/components: `dashboard/src/pages/WhatIfPage.jsx`, anomaly badge; `reports/traceability.md`, anomaly wording.  
   - Issue: UI says “Anomalous execution profile,” but the detector sees only pre-build engineered metadata, not execution duration/log telemetry. Its observed anomalous/normal failure rates are effectively equal (21.175% vs 21.161%).  
   - Why it matters: an examiner could reasonably challenge the claim that anomaly status is a risk signal or execution-profile analysis.  
   - Impact: documentation/demo quality, not classifier correctness.

4. **Stale documents contradict current implementation/state**  
   - Files: `reports/traceability.md`, `reports/demo_script.md`, `reports/optimization_simulator_design.md`, `reports/regression_diagnosis.md`.  
   - Issue: older statements claim data/artifacts are unavailable, retain outdated controls/response sign vocabulary, or contain the TP-count prose discrepancy.  
   - Why it matters: current code/test evidence is stronger than these documents, but contradictory evidence undermines viva confidence.  
   - Impact: documentation/viva quality.

### MINOR

1. **No frontend automated component/browser tests**  
   - Location: `dashboard/package.json`.  
   - Issue: there is no configured frontend test runner; current validation is Vite build plus lint.  
   - Impact: regression coverage; backend contracts are well tested.

2. **Production hardening remains demo-oriented**  
   - Files: `api/auth/security.py`, `api/db/init_db.py`, `api/db/models.py`, `api/main.py`.  
   - Issue: deliberate development-secret/default-password behavior, SQLite, global pipeline-name lookup, and localhost-only CORS are appropriate to a capstone demo but not multi-user deployment.  
   - Impact: deployment security/scalability, not local viva operation when real secrets are set.

3. **Frontend lint warning and bundle-size warning**  
   - Files: `dashboard/src/auth/AuthContext.jsx`; dashboard build output.  
   - Issue: Fast Refresh export warning and a >500 kB production chunk warning.  
   - Impact: cosmetic/development ergonomics; build succeeds.

### COSMETIC

None requiring action before the viva.

## 20. Historical Verdict

# READY WITH MINOR FIXES

The core ML, leakage safeguards, artifact wiring, SHAP, post-failure analysis, secured API workflows, and database-free simulations are internally consistent and tested. The Task 11 follow-up resolved the listed documentation, anomaly-language, root-cause, and manual-input concerns without retraining.

## 21. Current Status After Task 11

# READY FOR VIVA

The frozen Random Forest artifacts remain unchanged, the full backend suite has
34 passing tests, and the dashboard build/lint checks pass with the known Fast
Refresh warning only. Docker configuration is verified, while Docker runtime
remains unverified because daemon access was unavailable.

## 22. Completed Follow-Up

**Task 11 — Resolve final viva consistency findings:** completed. Manual
submission uses canonical file counts; root-cause analysis requires a non-blank
failed-test log; anomaly language distinguishes unusual pre-build input profiles
from failure risk; and final documentation is synchronized.
