# 5–10 Minute Viva Demo Script

Use the public demo, or run it locally.

**Run it locally:**

```bash
cd dashboard && VITE_API_BASE_URL=/api npm run build && cd ..
DATABASE_URL=sqlite:///./tmp/demo.db .venv/bin/python -m api.db.init_db
DATABASE_URL=sqlite:///./tmp/demo.db .venv/bin/python scripts/seed_demo.py
DEMO_READ_ONLY=1 DATABASE_URL=sqlite:///./tmp/demo.db .venv/bin/uvicorn api.demo:app --port 8000
```

**About the data.** The demo holds 450 real GitHub Actions runs from three
public repositories. They come from the model's holdout window, so every score
shown is out-of-sample.

## Walkthrough (about 7 minutes)

1. **0:00–1:00 — the claim.** Open the README's "Results at a glance" table.
   - Tier E beat Tier C on Travis and on two pre-registered GitHub Actions
     tests: 0.911 vs 0.763 within-repository AUC on 30 new repositories.
   - Explain "pre-registered": the protocol was hashed before the data was
     collected.
2. **1:00–2:00 — dashboard.** Walk through:
   - the failure trend;
   - the risk distribution ("each repository flags its riskiest 15%");
   - the model panel with its holdout numbers.
3. **2:00–3:30 — a repository.** Open Pipelines, then `yt-dlp/yt-dlp`.
   - Show the risk-over-time chart: the dots are real outcomes, and the dashed
     line is this repository's alert threshold.
   - Open one build's "Why" chart. Point out that history on the same branch or
     PR, and whether the previous run of this workflow failed, dominate.
4. **3:30–4:30 — how it runs live.** Explain the webhook:
   - GitHub sends `workflow_run` *requested*, and the run is scored before it
     executes;
   - *completed* attaches the real outcome, which becomes history;
   - nothing blocks CI.
   - Mention that the live history is tested equal to the training pass.
5. **4:30–5:30 — Submit.** Score a run on `yt-dlp/yt-dlp` with the head
   branch of a failing stream.
   - Show the history panel and the repository's threshold.
   - In the demo it is scored, not saved.
6. **5:30–6:30 — What-if.** Toggle "previous run on this branch failed" and
   watch the risk move.
   - Run one optimization simulation and read the non-causal disclaimer.
7. **6:30–7:00 — what's next.** Explain the forward-looking test: the deployed
   model and alert rule are frozen by hash and will be scored on October 2026
   runs in late November (`reports/phase2_prospective_protocol.md`).

## Before presenting

- Open the demo once a few minutes ahead. The free hosting plan sleeps when
  idle, and the first load takes about a minute.
- Time one rehearsal and record it here.
