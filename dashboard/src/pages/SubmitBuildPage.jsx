import { useState } from 'react'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { RiskBar } from '../components/charts'
import { PageHeader, ScrambleText } from '../components/ui'

// One GitHub Actions workflow run, described the way the deployed model was trained.
const initialForm = {
  pipeline: 'acme/api',
  workflowId: 101,
  isPr: true,
  author: 'octocat',
  branch: 'feature-login',
  additions: 40,
  deletions: 10,
  filesModified: 3,
  message: 'Fix login redirect',
  isMerge: false,
  outcome: '',
}

// Human-readable view of the history the server derived for this build.
function HistoryContext({ context }) {
  if (!context) return null
  const value = (key, format = v => v) => (context[key] === null || context[key] === undefined ? '—' : format(context[key]))
  const outcome = v => (v ? 'failed' : 'passed')
  const rate = v => `${(v * 100).toFixed(0)}%`
  const rows = [
    ['Finished runs on this branch/PR', value('hist_stream_completed')],
    ['Last outcome on this branch/PR', value('hist_stream_prev_failed', outcome)],
    ['Last run of this workflow here', value('hist_prevbuild_failed', outcome)],
    ['Pipeline failure rate (last 5)', value('hist_repo_fail_rate_5', rate)],
    ['Runs still in progress in repository', value('ctx_repo_pending')],
  ]
  return <div className="mt-4 border-t border-line pt-4">
    <h3 className="label font-sans">History used</h3>
    <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
      {rows.map(([label, shown]) => <div key={label}><dt className="text-xs text-ink-faint">{label}</dt><dd className="readout text-ink">{shown}</dd></div>)}
    </dl>
    {context.hist_repo_fail_rate_all === null && <p className="mt-3 rounded-control border border-warn/30 bg-warn/10 p-3 text-xs text-warn">
      No recorded outcomes for this repository yet, so the score rests on the commit and timing alone. History is the model's strongest signal: submit runs with their observed outcome to build it up.
    </p>}
  </div>
}

export default function SubmitBuildPage() {
  const [form, setForm] = useState(initialForm)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const change = event => {
    const { name, type, checked, value } = event.target
    setForm({ ...form, [name]: type === 'checkbox' ? checked : value })
  }

  const submit = async event => {
    event.preventDefault()
    setError('')
    try {
      const { data } = await client.post(`/pipelines/${encodeURIComponent(form.pipeline)}/builds?explain=true`, {
        build_features: {
          event: form.isPr ? 'pull_request' : 'push',
          workflow_id: Number(form.workflowId),
          head_branch: form.branch,
          // A pull request is tracked by its author and branch, as in training.
          ...(form.isPr ? { actor: form.author } : {}),
          additions: Number(form.additions),
          deletions: Number(form.deletions),
          files_modified: Number(form.filesModified),
          commit_message: form.message,
          num_parents: form.isMerge ? 2 : 1,
        },
        // A known outcome becomes history for later builds; without one only the prediction is stored.
        ...(form.outcome ? { build_status: form.outcome } : {}),
      })
      setResult(data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Build submission failed.')
    }
  }

  return (
    <>
      <PageHeader title="Submit build">Score a GitHub Actions run's failure risk before it runs. Connected repositories are scored automatically through the webhook.</PageHeader>
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,36rem)_minmax(0,1fr)]">
        <form onSubmit={submit} className="card grid gap-5 p-5 sm:p-6">
          <div className="grid gap-3 sm:grid-cols-[1fr_9rem]">
            <label className="label">Repository<input name="pipeline" value={form.pipeline} onChange={change} className="field" /></label>
            <label className="label">Workflow ID<input name="workflowId" type="number" min="1" value={form.workflowId} onChange={change} className="field" /></label>
          </div>
          <label className="flex items-center gap-2.5 text-sm text-ink"><input name="isPr" type="checkbox" checked={form.isPr} onChange={change} className="size-4" />Pull request run</label>
          <div className={`grid gap-3 ${form.isPr ? 'sm:grid-cols-2' : ''}`}>
            {form.isPr && <label className="label">PR author<input name="author" value={form.author} onChange={change} className="field" /></label>}
            <label className="label">{form.isPr ? 'Head branch' : 'Branch'}<input name="branch" value={form.branch} onChange={change} className="field" /></label>
          </div>
          <fieldset className="grid grid-cols-3 gap-3">
            <legend className="col-span-3 mb-1 text-sm font-medium text-ink">Head commit</legend>
            <label className="label">Lines added<input name="additions" type="number" min="0" value={form.additions} onChange={change} className="field" /></label>
            <label className="label">Lines deleted<input name="deletions" type="number" min="0" value={form.deletions} onChange={change} className="field" /></label>
            <label className="label">Files changed<input name="filesModified" type="number" min="0" value={form.filesModified} onChange={change} className="field" /></label>
          </fieldset>
          <label className="label">Commit message<input name="message" value={form.message} onChange={change} className="field" /></label>
          <label className="flex items-center gap-2.5 text-sm text-ink"><input name="isMerge" type="checkbox" checked={form.isMerge} onChange={change} className="size-4" />Merge commit</label>
          <label className="label">Observed outcome<select name="outcome" value={form.outcome} onChange={change} className="field"><option value="">Not known yet</option><option value="passed">Passed</option><option value="failed">Failed</option></select></label>
          <p className="-mt-3 text-xs text-ink-faint">Branch/PR history and in-progress runs come from this repository's recorded runs.</p>
          <button className="btn btn-primary">Analyze build</button>
        </form>
        <div>
          {error && <p className="rounded-control border border-danger/30 bg-danger/10 p-4 text-danger" role="alert">{error}</p>}
          {result && <section className="card tick animate-rise p-5 sm:p-6" aria-live="polite"><h2 className="text-xl text-ink"><ScrambleText text="Prediction result" /></h2><div className="mt-4 grid grid-cols-2 gap-3"><div className="inset-panel p-4"><p className="label">Predicted</p><p className="mt-2"><StatusBadge status={result.predicted_status} /></p></div><div className="inset-panel p-4"><p className="label">Failure probability</p><p className="readout mt-1 text-2xl">{(result.failure_probability * 100).toFixed(1)}%</p></div></div><RiskBar probability={result.failure_probability} threshold={result.decision_threshold} label={result.alert_basis === 'repository_budget' ? "this repository's alert threshold, set by its riskiest 15% of recent runs" : "a typical repository's alert threshold, used until this one has 20 scored runs"} /><HistoryContext context={result.stream_context} /></section>}
          {!result && !error && <div className="hidden rounded-card border border-dashed border-line-strong p-6 text-sm text-ink-muted lg:block">Run an analysis to see the predicted status and failure probability here.</div>}
        </div>
      </div>
    </>
  )
}
