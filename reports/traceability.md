# Report-to-Implementation Traceability

## Scope note

The referenced project report was not attached or found in this repository during
Task 15. This table therefore traces the Functional Requirements and five-layer
architecture described in the supplied implementation plan, not unseen text from
Chapters 4.1 and 5.1. Reconcile wording and add any report-only requirement before
presenting.

## Controlled canonical file-count retrain (Task 2)

The canonical mapping is `gh_diff_files_added + gh_diff_files_deleted +
gh_diff_files_modified`. A single controlled retrain/evaluation used this
mapping without changing the prediction schema, hyperparameters, algorithms, or
split. Random Forest remained the selected model with precision 0.3286, recall
0.6217, F1 0.4300, and AUC 0.6902. The anomaly detector was regenerated because
it consumes the same imputed prediction feature matrix; root-cause clustering
was not regenerated because it uses failed-test log text only.

| Report Requirement | Implementing Module/Endpoint | Verified (Y/N) |
|---|---|---|
| Ingest TravisTorrent CI/CD build data | `src/data/ingest.py`, `src/data/run_pipeline.py` | Y — real processed TravisTorrent data is present and used by the controlled retrain. |
| Clean passed/failed builds and derive time features | `src/data/preprocess.py`; `tests/test_preprocess.py` | Y |
| Leak-free pre-build failure features | `src/features/schema.py`, `src/features/engineer.py`; `tests/test_features.py` | Y — `tr_log_*` fields are excluded from the canonical feature schema. |
| Train RF, Decision Tree, SVM, and XGBoost models | `src/models/train.py` | Y — real-data artifacts were trained once from the reproducible 200,000-row cap after the canonical file-count correction. |
| Evaluate models and publish comparative metrics/charts | `src/models/evaluate.py`, `tests/test_evaluate.py` | Y — current metrics, CSV, best-model marker, and report charts were generated from the controlled retrain. |
| Detect anomalous builds | `src/models/anomaly.py`, `src/models/train_anomaly.py`, `tests/test_anomaly.py` | Y — regenerated against corrected prediction inputs; anomalous builds did not show a higher failure rate, so treat this as an unusual pre-build input-profile signal rather than a failure-risk signal. |
| Diagnose failed-build root causes from failed test names | `src/models/clustering.py`, `src/models/train_clustering.py`, `tests/test_clustering.py` | Y — real-data cluster artifact/chart generated; human review of representative cluster members remains recommended. |
| Predict a live build failure | `src/inference/predictor.py`, `tests/test_predictor.py` | Y — selected Random Forest and all supporting real-data artifacts load successfully. |
| Persist users, pipelines, logs, predictions, anomalies, and root-cause output | `api/db/models.py`, `api/db/database.py`; `tests/test_db.py` | Y |
| Authenticate developers and administrators with JWT and roles | `api/auth/security.py`, `api/auth/dependencies.py`, `api/routers/auth.py`; `tests/test_auth.py` | Y |
| Submit and view build predictions/dashboard data | `api/routers/pipelines.py`, `api/schemas.py`; `tests/test_pipelines.py`, `tests/test_e2e.py` | Y — the production Random Forest and supporting artifacts are present; API routes also have deterministic integration coverage. |
| Ingest GitHub Actions and Jenkins CI events | `api/routers/webhooks.py`, `scripts/simulate_ci_traffic.py`; `tests/test_webhooks.py`, `tests/test_webhook_signature.py` | Y for protected endpoint flow and signature validation; a public CI-provider delivery rehearsal remains optional. |
| Visualization Layer dashboard | `dashboard/src/pages/`, `dashboard/src/components/`; `npm run build` | Y — production build succeeds and the dashboard consumes the live API contracts. |
| Containerized deployment | `docker/Dockerfile.api`, `docker/Dockerfile.dashboard`, `docker-compose.yml` | N — Compose configuration validates, but Docker image build/run was blocked because the local Docker daemon was not running. |
| Data Collection Layer | `src/data/`, `api/routers/webhooks.py`, `scripts/simulate_ci_traffic.py` | Y — the real TravisTorrent raw export and processed Parquet dataset are present in this workspace. |
| Feature Engineering Layer | `src/features/` | Y |
| Prediction and Anomaly Detection Layer | `src/models/train.py`, `src/models/anomaly.py`, `src/inference/predictor.py` | Y — models and anomaly artifact are trained and loadable. |
| Root Cause Analysis Layer | `src/models/clustering.py`, `src/models/train_clustering.py` | Y — cluster artifact and chart are generated; review representative failures before making semantic cluster claims. |
| Visualization and API Layer | `api/`, `dashboard/` | Y in tests/build and real artifacts are available; Docker/server runtime validation remains pending because the Docker daemon was unavailable. |
