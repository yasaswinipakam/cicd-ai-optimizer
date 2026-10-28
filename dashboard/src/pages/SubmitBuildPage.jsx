import { useState } from 'react'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { useModelInfo } from '../api/model'
import { RiskBar } from '../components/charts'
import { PageHeader, ScrambleText } from '../components/ui'

// The languages the deployed model was trained on; any other value would be scored as unknown.
const LANGUAGES = ['go', 'java', 'python', 'ruby']

const initialForm = {
  pipeline: 'demo-repo',
  src: 50,
  filesAdded: 1,
  filesDeleted: 1,
  filesModified: 4,
  language: 'ruby',
  isPr: true,
  prNumber: 1,
  branch: 'main',
  outcome: '',
}

// Human-readable view of the history the server derived for this build.
function HistoryContext({ context }) {
  if (!context) return null
  const value = (key, format = v => v) => (context[key] === null || context[key] === undefined ? '—' : format(context[key]))
  const outcome = v => (v ? 'failed' : 'passed')
  const rate = v => `${(v * 100).toFixed(0)}%`
  const rows = [
    ['Finished builds on this branch/PR', value('hist_stream_completed')],
    ['Last outcome on this branch/PR', value('hist_stream_prev_failed', outcome)],
    ['Pipeline failure rate (last 5)', value('hist_repo_fail_rate_5', rate)],
    ['Builds still running in pipeline', value('ctx_repo_pending')],
  ]
  return <div className="mt-4 border-t border-line pt-4">
    <h3 className="label font-sans">History used</h3>
    <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
      {rows.map(([label, shown]) => <div key={label}><dt className="text-xs text-ink-faint">{label}</dt><dd className="readout text-ink">{shown}</dd></div>)}
    </dl>
    {context.hist_repo_fail_rate_all === null && <p className="mt-3 rounded-control border border-warn/30 bg-warn/10 p-3 text-xs text-warn">
      No recorded outcomes for this pipeline yet. The model treats it like a brand-new CI setup, whose first builds failed about half the time in training, so risk reads high. Submit builds with their observed outcome to build up history.
    </p>}
  </div>
}

export default function SubmitBuildPage() {
  const [form, setForm] = useState(initialForm)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const model = useModelInfo()

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
          git_diff_src_churn: Number(form.src),
          gh_diff_files_added: Number(form.filesAdded),
          gh_diff_files_deleted: Number(form.filesDeleted),
          gh_diff_files_modified: Number(form.filesModified),
          gh_lang: form.language,
          gh_is_pr: form.isPr,
          ...(form.isPr && form.prNumber !== '' ? { gh_pull_req_num: Number(form.prNumber) } : {}),
          git_branch: form.branch,
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
      <PageHeader title="Submit build">Score a build's failure risk from its pre-build change profile.</PageHeader>
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,36rem)_minmax(0,1fr)]">
        <form onSubmit={submit} className="card grid gap-5 p-5 sm:p-6">
          <label className="label">Pipeline ID<input name="pipeline" value={form.pipeline} onChange={change} className="field" /></label>
          <label className="label">Source churn<input name="src" type="number" value={form.src} onChange={change} className="field" /></label>
          <fieldset className="grid grid-cols-3 gap-3">
            <legend className="col-span-3 mb-1 text-sm font-medium text-ink">Changed files</legend>
            <label className="label">Added<input name="filesAdded" type="number" min="0" value={form.filesAdded} onChange={change} className="field" /></label>
            <label className="label">Deleted<input name="filesDeleted" type="number" min="0" value={form.filesDeleted} onChange={change} className="field" /></label>
            <label className="label">Modified<input name="filesModified" type="number" min="0" value={form.filesModified} onChange={change} className="field" /></label>
          </fieldset>
          <label className="label">Language<select name="language" value={form.language} onChange={change} className="field">{LANGUAGES.map(language => <option key={language}>{language}</option>)}</select></label>
          <label className="label">Branch<input name="branch" value={form.branch} onChange={change} className="field" /></label>
          <label className="flex items-center gap-2.5 text-sm text-ink"><input name="isPr" type="checkbox" checked={form.isPr} onChange={change} className="size-4" />Pull request build</label>
          {form.isPr && <label className="label">PR number<input name="prNumber" type="number" min="1" value={form.prNumber} onChange={change} className="field" /></label>}
          <label className="label">Observed outcome<select name="outcome" value={form.outcome} onChange={change} className="field"><option value="">Not known yet</option><option value="passed">Passed</option><option value="failed">Failed</option></select></label>
          <p className="-mt-3 text-xs text-ink-faint">Branch/PR history and running builds come from this pipeline's recorded builds.</p>
          <button className="btn btn-primary">Analyze build</button>
        </form>
        <div>
          {error && <p className="rounded-control border border-danger/30 bg-danger/10 p-4 text-danger" role="alert">{error}</p>}
          {result && <section className="card tick animate-rise p-5 sm:p-6" aria-live="polite"><h2 className="text-xl text-ink"><ScrambleText text="Prediction result" /></h2><div className="mt-4 grid grid-cols-2 gap-3"><div className="inset-panel p-4"><p className="label">Predicted</p><p className="mt-2"><StatusBadge status={result.predicted_status} /></p></div><div className="inset-panel p-4"><p className="label">Failure probability</p><p className="readout mt-1 text-2xl">{(result.failure_probability * 100).toFixed(1)}%</p></div></div><RiskBar probability={result.failure_probability} threshold={model?.decision_threshold} /><HistoryContext context={result.stream_context} /></section>}
          {!result && !error && <div className="hidden rounded-card border border-dashed border-line-strong p-6 text-sm text-ink-muted lg:block">Run an analysis to see the predicted status and failure probability here.</div>}
        </div>
      </div>
    </>
  )
}
