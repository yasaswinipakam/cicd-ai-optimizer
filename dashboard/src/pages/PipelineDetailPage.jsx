import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'

function ExplanationChart({ explanation }) {
  if (!explanation?.length) return null
  const data = explanation.map(item => ({ ...item, magnitude: Math.abs(item.shap_value) }))
  return <div className="mt-2 h-40 rounded bg-slate-50 p-2"><p className="mb-1 text-xs font-semibold">Why this prediction</p><ResponsiveContainer><BarChart data={data} layout="vertical" margin={{ left: 35 }}><XAxis type="number" hide /><YAxis type="category" dataKey="feature" width={110} tick={{ fontSize: 10 }} /><Tooltip formatter={(value, _name, props) => [props.payload.shap_value.toFixed(3), 'SHAP value']} /><Bar dataKey="magnitude">{data.map((entry, index) => <Cell key={index} fill={entry.direction === 'increases_failure_risk' ? '#dc2626' : '#2563eb'} />)}</Bar></BarChart></ResponsiveContainer></div>
}

function RootCause({ build }) {
  if (build.root_cause_label) return <>{build.root_cause_label}</>
  if (build.build_status === 'failed' || build.predicted_status === 'failed') {
    return <span className="text-slate-500">No failed-test log available for root-cause analysis.</span>
  }
  return <span className="text-slate-400">—</span>
}

function TimingBadge({ status }) {
  const classes = {
    normal: 'bg-emerald-100 text-emerald-800',
    bottleneck: 'bg-amber-100 text-amber-900',
    stalled: 'bg-red-100 text-red-800',
    awaiting_start: 'bg-slate-100 text-slate-600',
    missing_timestamps: 'bg-slate-100 text-slate-600',
    insufficient_history: 'bg-slate-100 text-slate-600',
  }
  return <span className={`rounded px-2 py-1 text-xs font-medium ${classes[status] || classes.insufficient_history}`}>{status?.replaceAll('_', ' ') || 'unknown'}</span>
}

function JobTelemetry({ jobs, analyses, onShowAnalysis }) {
  if (!jobs.length) return <p className="rounded border border-dashed border-slate-300 p-4 text-sm text-slate-500">No GitHub Actions job telemetry has arrived for this pipeline yet.</p>
  const seconds = value => value == null ? '—' : `${Math.round(value)}s`
  return <div className="overflow-hidden rounded-lg bg-white shadow-sm"><table className="w-full text-left text-sm"><thead className="bg-slate-100 text-slate-600"><tr><th className="p-3">Job</th><th>Completed</th><th>Duration</th><th>Historical p95</th><th>Runtime assessment</th><th>Failed-job diagnostics</th></tr></thead><tbody>{jobs.map(job => {
    const analysis = analyses[job.job_run_id]
    const failed = ['failure', 'timed_out', 'cancelled', 'action_required'].includes(job.conclusion)
    return <tr key={job.job_run_id} className="border-t align-top"><td className="p-3"><p className="font-medium">{job.job_name}</p><p className="text-xs text-slate-500">{job.workflow_name || 'Workflow'}</p></td><td>{job.completed_at ? new Date(job.completed_at).toLocaleString() : job.status}</td><td>{seconds(job.duration_seconds)}</td><td>{seconds(job.baseline_p95_seconds)}<p className="text-xs text-slate-500">n={job.baseline_sample_size}</p></td><td><TimingBadge status={job.bottleneck_status} /></td><td className="py-3 pr-3">{failed ? <>{analysis?.loading ? <span className="text-slate-500">Loading…</span> : analysis?.error ? <span className="text-slate-500">No captured log yet.</span> : analysis ? <div className="rounded bg-orange-50 p-2 text-xs"><b>{analysis.title}</b><p className="mt-1 text-slate-700">{analysis.summary}</p>{analysis.matched_signals?.length > 0 && <p className="mt-1 break-words font-mono text-slate-600">{analysis.matched_signals[0]}</p>}</div> : <button onClick={() => onShowAnalysis(job.job_run_id)} className="rounded bg-orange-700 px-2 py-1 text-xs text-white hover:bg-orange-800">View diagnosis</button>}</> : <span className="text-slate-400">—</span>}</td></tr>
  })}</tbody></table></div>
}

export default function PipelineDetailPage() {
  const { pipelineId } = useParams()
  const [builds, setBuilds] = useState([])
  const [jobs, setJobs] = useState([])
  const [analyses, setAnalyses] = useState({})

  useEffect(() => {
    const encoded = encodeURIComponent(pipelineId)
    client.get(`/pipelines/${encoded}/builds`).then(response => setBuilds(response.data)).catch(() => setBuilds([]))
    client.get(`/pipelines/${encoded}/telemetry/jobs`).then(response => setJobs(response.data)).catch(() => setJobs([]))
  }, [pipelineId])

  const showFailureAnalysis = async jobId => {
    setAnalyses(current => ({ ...current, [jobId]: { loading: true } }))
    try {
      const response = await client.get(`/pipelines/${encodeURIComponent(pipelineId)}/telemetry/jobs/${jobId}/failure-analysis`)
      setAnalyses(current => ({ ...current, [jobId]: response.data }))
    } catch {
      setAnalyses(current => ({ ...current, [jobId]: { error: true } }))
    }
  }

  const downloadReport = async build => {
    const response = await client.get(`/pipelines/${encodeURIComponent(pipelineId)}/builds/${build.id}/report.pdf`, { responseType: 'blob' })
    const url = URL.createObjectURL(response.data)
    const link = document.createElement('a')
    link.href = url
    link.download = `build_${build.id}_report.pdf`
    link.click()
    URL.revokeObjectURL(url)
  }

  return <><h1 className="mb-2 text-2xl font-bold">{pipelineId}</h1><p className="mb-5 text-sm text-slate-600">Pre-build risk scoring is separate from the runtime telemetry and post-failure evidence below.</p><section className="mb-8"><h2 className="mb-3 text-lg font-semibold">Runtime job telemetry and bottlenecks</h2><JobTelemetry jobs={jobs} analyses={analyses} onShowAnalysis={showFailureAnalysis} /></section><section><h2 className="mb-3 text-lg font-semibold">Pre-build predictions and completed-build history</h2><div className="overflow-hidden rounded-lg bg-white shadow-sm"><table className="w-full text-left text-sm"><thead className="bg-slate-100 text-slate-600"><tr><th className="p-3">Timestamp</th><th>Status</th><th>Failure probability</th><th>Input-profile anomaly</th><th>Root cause, optimization & explanation</th><th>Report</th></tr></thead><tbody>{builds.map(build => <tr key={build.id} className="border-t align-top"><td className="p-3">{new Date(build.timestamp).toLocaleString()}</td><td><StatusBadge status={build.build_status} /></td><td>{build.failure_probability == null ? '—' : `${(build.failure_probability * 100).toFixed(1)}%`}</td><td className="py-3 pr-3">{build.is_anomaly ? <><span>Unusual pre-build profile</span><span className="mt-1 block text-xs text-slate-500">Not a failure prediction</span></> : 'No unusual profile'}</td><td className="py-3 pr-3"><RootCause build={build} />{build.optimization_suggestion && <div className="mt-2 rounded bg-sky-50 p-2 text-xs text-slate-700"><b>{build.optimization_suggestion.category}: {build.optimization_suggestion.title}</b><p className="mt-1">{build.optimization_suggestion.suggestion}</p></div>}<ExplanationChart explanation={build.explanation} /></td><td className="py-3 pr-3"><button onClick={() => downloadReport(build)} className="rounded bg-sky-700 px-2 py-1 text-xs text-white hover:bg-sky-800">Download Report</button></td></tr>)}</tbody></table></div></section></>
}
