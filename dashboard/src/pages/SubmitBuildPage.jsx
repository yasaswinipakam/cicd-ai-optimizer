import { useState } from 'react'
import client from '../api/client'

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
      <h1 className="mb-5 text-2xl font-bold">Submit build</h1>
      <form onSubmit={submit} className="grid max-w-xl gap-4 rounded-lg bg-white p-5 shadow-sm">
        <label>Pipeline ID<input name="pipeline" value={form.pipeline} onChange={change} className="mt-1 w-full rounded border p-2" /></label>
        <label>Source churn<input name="src" type="number" value={form.src} onChange={change} className="mt-1 w-full rounded border p-2" /></label>
        <fieldset className="grid grid-cols-3 gap-4">
          <legend className="col-span-3 text-sm font-medium text-slate-700">Changed files</legend>
          <label>Added<input name="filesAdded" type="number" min="0" value={form.filesAdded} onChange={change} className="mt-1 w-full rounded border p-2" /></label>
          <label>Deleted<input name="filesDeleted" type="number" min="0" value={form.filesDeleted} onChange={change} className="mt-1 w-full rounded border p-2" /></label>
          <label>Modified<input name="filesModified" type="number" min="0" value={form.filesModified} onChange={change} className="mt-1 w-full rounded border p-2" /></label>
        </fieldset>
        <p className="-mt-2 text-xs text-slate-500">Total files changed is derived from added + deleted + modified.</p>
        <label>Language<select name="language" value={form.language} onChange={change} className="mt-1 w-full rounded border p-2"><option>ruby</option><option>python</option><option>java</option><option>javascript</option></select></label>
        <label>Branch<input name="branch" value={form.branch} onChange={change} className="mt-1 w-full rounded border p-2" /></label>
        <label className="flex gap-2"><input name="isPr" type="checkbox" checked={form.isPr} onChange={change} />Pull request build</label>
        <button className="rounded bg-sky-700 p-2 font-medium text-white">Analyze build</button>
      </form>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {result && <section className="mt-6 rounded-lg bg-white p-5 shadow-sm"><h2 className="font-semibold">Prediction result</h2><p className="mt-2">Predicted: <b>{result.predicted_status}</b> · failure probability: <b>{(result.failure_probability * 100).toFixed(1)}%</b></p><p>Input-profile anomaly: {result.is_anomaly ? 'Unusual pre-build profile (not a failure prediction)' : 'No unusual profile detected'} · Root cause: {result.root_cause_label || (result.predicted_status === 'failed' ? 'No failed-test log available for root-cause analysis.' : '—')}</p></section>}
    </>
  )
}
