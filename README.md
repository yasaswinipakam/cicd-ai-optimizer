# AI-Driven Analysis and Optimization of CI/CD Pipelines for Build Failure Prediction

This project analyzes CI/CD build data to predict build failures and support pipeline optimization. It combines a Python machine-learning workflow, a FastAPI service, and a React dashboard, with room for reproducible artifacts, reports, and containerized deployment.

## Current validated model

The deployed failure model is the **Tier E blend** from the Phase-1 study
(`artifacts/tier_e_blend.joblib`, built by `src.models.train_tier_e`).
It blends a binary LightGBM with a LightGBM ranker, standardizing each
model's scores with training-score moments. A global Platt map turns the
blended score into a probability.

**Inputs.** The model reads the build's change profile plus history for its
*stream* (the pull request, or the branch for push builds) and its in-flight
context. In-flight context means builds in the same pipeline that are still
running. The API derives this history from the pipeline's stored builds.
Clients send `git_branch`, `gh_is_pr` and `gh_pull_req_num`, and report each
build's outcome. Only reported outcomes become history; builds awaiting an
outcome count as running.

**Holdout results.** The deployed model is exactly the walk-forward fold-10
model: trained on chronological folds 1–9 and tested on fold 10, which it
reproduces bit for bit. On that holdout:

| Measure | Value |
|---|---:|
| Within-repo AUC | 0.690 |
| Pooled AUC | 0.844 |
| Prevalence | 18.5% |

The alert threshold is set so that 15% of holdout builds are flagged, a
review budget chosen without using labels. At that threshold, precision is
0.71 and recall 0.57. The headline across folds 7–10 is 0.695 within-repo
AUC, or about 0.62 on new breakages; see
[`reports/final_results.md`](reports/final_results.md).

**Cold start.** A pipeline with no recorded outcomes scores high. In
training, a repository's first builds failed about half the time. Record a
few outcomes before reading the scores.

**Random Forest.** The 39-feature Random Forest below still provides the
anomaly-detection matrix and the what-if derived features. Rerunning
`src.models.train --mode full` points `best_model.txt` back at it.

The legacy deployable model is a Random Forest using 34 availability-guarded
pre-build features. Its prior calibration experiment fitted the model on the
earliest 72% of the 3.56M job rows and sigmoid/Platt scaling on the following
8%. Although the cutoff itself was validation-derived, the recorded activation
rule also inspected final-20% test metrics; consequently that final period must
be treated as a labelled development test, not an untouched confirmation set.
The active-artifact metadata currently records an uncalibrated full model with a
0.40 threshold, while a separate sigmoid calibrator artifact and historical
0.287911 policy report also exist. Do not treat either as an active calibrated
policy until the Phase-1 versioned evaluation reconciles them. See
[`reports/chronological_calibration_analysis.md`](reports/chronological_calibration_analysis.md).

`fast` mode retains the 200,000-row stratified random-split workflow only for
development and four-model comparison. The notebook's stronger random-split
scores are not deployment metrics; earlier scores that included post-build log
features were leakage-inflated.

The classifier uses pre-build inputs only. Anomaly status means an unusual
pre-build input profile, not a prediction that a build will fail. Root-cause
clusters and suggestions use failed-test-log text after a build fails.

## Setup

Create and activate a Python 3.11 virtual environment, then install the project dependencies:

```bash
cd cicd-ai-optimizer
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```

Place `final-2017-01-25_csv.gz` in `data/raw/` when it is available.

## Prepare model artifacts

The application needs trained artifacts before live prediction, anomaly detection,
root-cause analysis, and the fully populated dashboard are available. With the raw
dataset in place, run the following from the project root:

```bash
.venv/bin/python -m src.data.run_pipeline --input data/raw/final-2017-01-25_csv.gz --output data/processed/clean.parquet

# Fast: 200k random-split four-model development comparison.
.venv/bin/python -m src.models.train --mode fast

# Full: deployable Random Forest trained on earlier history and evaluated on later history.
.venv/bin/python -m src.models.train --mode full

# Run the strictly chronological calibration study. It activates a candidate
# only if it improves reliability AND preserves the same base model's 15% tier.
.venv/bin/python -m src.models.chronological_calibration

# Rebuild dependent anomaly artifact after the persisted preprocessing changes.
.venv/bin/python -m src.models.train_anomaly
.venv/bin/python -m src.models.train_clustering

# Deploy the Tier E blend (run last: it repoints best_model.txt). It checks that
# it reproduces the saved research fold-10 predictions in reports/phase1_v2_parts/.
.venv/bin/python -m src.models.train_tier_e
```

`src.models.evaluate` is intended for the `fast` four-model comparison artifacts.
The full-mode temporal metrics are written to
`reports/enhanced_full_temporal_evaluation.csv` and `.md`.

## Alert policy and shadow deployment

The deployed full-mode model is a **risk-ranking and developer-assistance** tool;
it must not automatically block CI builds. The fixed-volume policy analysis is in
[`reports/full_temporal_alert_policy.md`](reports/full_temporal_alert_policy.md).
On the untouched temporal test period, the predeclared 5%, 10%, 15%, 20%, and 35%
top-risk policies provide explicit alert-volume versus failure-capture trade-offs.

Use the authenticated shadow endpoints to record a prediction before a CI result
exists, then attach the observed result later. They never change CI execution:

```text
POST  /pipelines/{pipeline_id}/shadow-builds
PATCH /pipelines/{pipeline_id}/shadow-builds/{build_id}/outcome
```

Each shadow record persists the risk score, decision threshold, model artifact
SHA-256, calibration method/calibrator SHA-256, feature schema, model training
mode, timestamp, pipeline, and eventual actual status. Generate and activate the
calibrated review policy before starting shadow collection:

```bash
.venv/bin/python -m src.models.chronological_calibration
```

The active sigmoid-calibrated cutoff is `0.287911`, selected from the
chronological validation window. It produces a 15.56% alert rate, 35.6% tier
precision, and captures 28.2% of failures on the untouched final period. The
earlier 80%-training raw model's 15% result is not a calibration-only comparator
because it uses a different training window. Monitor precision/recall, alert
volume, calibration by month/repository/language, and do not enable automatic
blocking.

After completed shadow outcomes accumulate, generate the read-only monitoring
breakdowns (overall, month, repository, and language) with:

```bash
.venv/bin/python -m src.models.shadow_monitoring
```

Initialize the local database with `.venv/bin/python -m api.db.init_db`, then run
the API with `.venv/bin/uvicorn api.main:app --reload` and the dashboard with
`cd dashboard && npm run dev`.

## Run with Docker

Copy the environment template and replace every placeholder with a real secret:

```bash
cp .env.example .env
docker compose up --build
```

Visit http://localhost:5173 to open the dashboard. The API is available at
http://localhost:8000. Build model artifacts first using the preceding section;
they are mounted into the API container. Stop the stack with `docker compose down`.

## GitHub webhook security

Set `GITHUB_WEBHOOK_SECRET` to exactly the same value configured in the GitHub
repository's **Settings -> Webhooks -> Secret** field. The API validates GitHub's
`X-Hub-Signature-256` HMAC signature against the original request bytes before
parsing or persisting an event. See GitHub's [webhook delivery validation guide](https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries).

## Failed-job log diagnostics collector

The signed `workflow_job` webhook supplies timing metadata, but not console log
text. To enable real post-failure diagnostic classification, add the included
[`ci-diagnostics-log-collector.yml`](.github/workflows/ci-diagnostics-log-collector.yml)
to the monitored GitHub repository, change `workflows: ["CI"]` to its primary
CI workflow's display name, and set these repository Actions secrets:

- `CI_DIAGNOSTICS_URL`: public HTTPS base URL of this API, for example `https://ci.example.edu`.
- `CI_DIAGNOSTICS_COLLECTOR_SECRET`: exactly the server's `CI_DIAGNOSTICS_COLLECTOR_SECRET` value.

The collector uses the repository-scoped `GITHUB_TOKEN` with `actions: read` to
download failed job logs, then posts only to the post-failure diagnostics endpoint.
It does not score builds, change CI outcomes, or put log content into the
pre-build model. GitHub documents that its job-log endpoint returns a short-lived
redirect and requires Actions read access for private repositories; see the
[job-log API documentation](https://docs.github.com/en/rest/actions/workflow-jobs#download-job-logs-for-a-workflow-run).
