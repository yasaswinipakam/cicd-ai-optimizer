# Report-to-Implementation Traceability

How each requirement maps to code and tests in the current system. The headline
evidence is in [`final_results.md`](final_results.md).

| Requirement | Implementing module or endpoint | Verified by |
|---|---|---|
| Ingest CI build data | TravisTorrent: `src/data/ingest.py`, `src/data/run_pipeline.py`. GitHub Actions: `scripts/collect_gha_confirmation.py` | `tests/test_preprocess.py` |
| Leak-free pre-build features, enforced at runtime | `src/features/schema.py`, `src/features/engineer.py`, `src/features/availability_guard.py`, `reports/feature_availability_registry.json` | `tests/test_features.py`, `tests/test_availability_guard.py` |
| Stream history and in-flight context, identical live and in training | `src/features/stream_context.py`, `api/routers/pipelines.py` (`_with_stream_context`) | `tests/test_stream_context.py`, `tests/test_tier_e_gha.py` (the live context matches the training pass row by row) |
| Deployed failure model (Tier E, trained on GitHub Actions) | `src/models/tier_e.py`, `src/models/train_tier_e_gha.py`, `src/inference/predictor.py` | `tests/test_tier_e.py`, `tests/test_tier_e_gha.py`; the trainer reproduces the research fold-10 predictions exactly |
| Per-repository alert budget | `src/inference/alert_policy.py`, `GET /pipelines/{id}/alert-threshold` | `tests/test_alert_policy.py`, `tests/test_pipelines.py` |
| Score GitHub Actions runs automatically (shadow mode, never blocking) | `POST /webhooks/github-actions`: `requested` is scored, `completed` attaches the outcome | `tests/test_tier_e_gha.py`, `tests/test_webhooks.py`, `tests/test_webhook_signature.py` |
| Explain each prediction | `TierEBlend.contributions` (SHAP values of the binary half) | `tests/test_tier_e.py`, `tests/test_explainer.py` |
| Diagnose failures from logs (after the build, never a model input) | `src/diagnostics/log_analysis.py`, `src/models/clustering.py` | `tests/test_log_analysis.py`, `tests/test_clustering.py` |
| Store pipelines, builds and predictions | `api/db/models.py`, `api/db/migrations.py` | `tests/test_db.py`, `tests/test_init_db.py` |
| Dashboard: trends, risk distribution, pipelines, submit, what-if | `dashboard/src/pages/`, `api/routers/pipelines.py`, `api/routers/predict.py` | `tests/test_pipelines.py`, `tests/test_predict_whatif.py`, `tests/test_e2e.py`; `npm run build` |
| Per-build PDF report | `src/reports/pdf_generator.py` | `tests/test_pdf_report.py` |
| Containerised and public deployment | `docker-compose.yml`, `docker/Dockerfile.api`, `docker/Dockerfile.dashboard`; read-only demo: `docker/Dockerfile.demo`, `render.yaml`, `api/demo.py` | `tests/test_demo.py` |
| Pre-registered evaluation | `reports/phase1_v2_gha_*_protocol.md`, `reports/phase2_prospective_protocol.md` and their hash files | the recorded SHA-256s |
