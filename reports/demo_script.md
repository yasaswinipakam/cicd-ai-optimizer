# 5–10 Minute Viva Demo Script

## Pre-demo checklist (must be complete before the presentation)

1. Confirm `data/raw/final-2017-01-25_csv.gz`, `data/processed/clean.parquet`, and
   the frozen artifacts are present. Do **not** retrain for the presentation.
2. Confirm `reports/final_results.md` contains the current held-out Random Forest
   F1 of 0.4300 and AUC of 0.6902, and `reports/figures/` contains the generated PNGs.
3. Create `.env` from `.env.example` with real, non-placeholder secrets.
4. Start Docker Desktop, then run `docker compose up --build`.
5. In another terminal export the same `GITHUB_WEBHOOK_SECRET` and prepare the
   simulator command below. Open http://localhost:5173.

## Walkthrough (target: 7 minutes)

1. **0:00–0:45 — architecture and startup.** Show `docker compose up --build`
   and point out the API and dashboard services. Explain that GitHub Actions and
   Jenkins webhooks feed the data-collection layer.
2. **0:45–1:30 — authentication.** Log in as `admin` with the password configured
   in `.env`; show the Dashboard page and four live summary cards.
3. **1:30–2:30 — streaming CI data.** Run:

   ```bash
   .venv/bin/python scripts/simulate_ci_traffic.py --count 5 --endpoint github-actions
   ```

   Refresh the dashboard and show totals/trend data increasing. Mention the
   webhook secret guards this unauthenticated machine-to-machine endpoint.
4. **2:30–3:30 — pipeline detail.** Open Pipelines, choose a failed build, and
   explain predicted failure probability, the unusual pre-build input-profile
   flag (not a failure prediction), and post-hoc root-cause label. Clarify that
   test-log clustering is diagnostic after a failure, not a predictive feature.
5. **3:30–4:30 — explanation and recommendation.** Show the persisted local SHAP
   chart, the failed-build recommendation, and the downloadable PDF. Explain that
   SHAP attributes this model score; it does not prove causal effects.
6. **4:30–5:20 — manual prediction.** Use Submit Build with source churn and the
   canonical added/deleted/modified file counts. Show the returned probability and
   the new persisted build entry.
7. **5:20–6:10 — What-If and optimization.** Open What-If, change one raw input,
   then run one optimization simulation. Show baseline versus simulated risk and
   read the non-causal disclaimer.
8. **6:10–6:40 — model evidence.** Open `reports/figures/model_comparison.png`,
   precision-recall curves, and feature-importance charts. Quote only the exact
   metrics in `reports/final_results.md`; explain why its best-F1 model was chosen.
9. **6:40–7:00 — traceability and safeguards.** Open `reports/traceability.md`.
   Highlight the explicit exclusion of `tr_log_*` from pre-build prediction and
   the items marked N, if any remain.

## Timing record

This script has been reviewed for sequence and prerequisites, but it has **not
been timed end-to-end** in this workspace because Docker runtime access remains
unavailable. The real dataset and frozen model artifacts are present. Perform one
timed rehearsal after the pre-demo checklist; record the duration here before
presenting.
