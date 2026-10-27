import { useState } from 'react'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { PageHeader, ScrambleText } from '../components/ui'

const initialForm = {
  pipeline: 'demo-repo',
  src: 50,
  filesAdded: 1,
  filesDeleted: 1,
  filesModified: 4,
  language: 'ruby',
  isPr: true,
  branch: 'main',
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
      const { data } = await client.post(`/pipelines/${encodeURIComponent(form.pipeline)}/builds`, {
        git_diff_src_churn: Number(form.src),
        gh_diff_files_added: Number(form.filesAdded),
        gh_diff_files_deleted: Number(form.filesDeleted),
        gh_diff_files_modified: Number(form.filesModified),
        gh_lang: form.language,
        gh_is_pr: form.isPr,
        git_branch: form.branch,
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
          <p className="-mt-3 text-xs text-ink-faint">Total files changed is derived from added + deleted + modified.</p>
          <label className="label">Language<select name="language" value={form.language} onChange={change} className="field"><option>ruby</option><option>python</option><option>java</option><option>javascript</option></select></label>
          <label className="label">Branch<input name="branch" value={form.branch} onChange={change} className="field" /></label>
          <label className="flex items-center gap-2.5 text-sm text-ink"><input name="isPr" type="checkbox" checked={form.isPr} onChange={change} className="size-4" />Pull request build</label>
          <button className="btn btn-primary">Analyze build</button>
        </form>
        <div>
          {error && <p className="rounded-control border border-danger/30 bg-danger/10 p-4 text-danger" role="alert">{error}</p>}
          {result && <section className="card tick animate-rise p-5 sm:p-6" aria-live="polite"><h2 className="text-xl text-ink"><ScrambleText text="Prediction result" /></h2><div className="mt-4 grid grid-cols-2 gap-3"><div className="inset-panel p-4"><p className="label">Predicted</p><p className="mt-2"><StatusBadge status={result.predicted_status} /></p></div><div className="inset-panel p-4"><p className="label">Failure probability</p><p className="readout mt-1 text-2xl">{(result.failure_probability * 100).toFixed(1)}%</p></div></div><dl className="mt-4 space-y-3 text-sm"><div><dt className="label">Input-profile anomaly</dt><dd className="mt-1 text-ink">{result.is_anomaly ? 'Unusual pre-build profile (not a failure prediction)' : 'No unusual profile detected'}</dd></div><div><dt className="label">Root cause</dt><dd className="mt-1 text-ink">{result.root_cause_label || (result.predicted_status === 'failed' ? 'No failed-test log available for root-cause analysis.' : '—')}</dd></div></dl></section>}
          {!result && !error && <div className="hidden rounded-card border border-dashed border-line-strong p-6 text-sm text-ink-muted lg:block">Run an analysis to see the predicted status and failure probability here.</div>}
        </div>
      </div>
    </>
  )
}
