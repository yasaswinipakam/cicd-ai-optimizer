import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { PageHeader, ScrambleText, TableShell } from '../components/ui'
import { featureLabel } from '../components/featureLabels'
import { axisProps, chartColors, tooltipProps } from '../theme/chart'

function ExplanationChart({ explanation }) {
  if (!explanation?.length) return null
  const data = explanation.map(item => ({ ...item, label: featureLabel(item.feature), magnitude: Math.abs(item.shap_value) }))
  return <div className="inset-panel mt-3 h-44 p-3 pb-8"><p className="label mb-1 text-xs">Why this prediction</p><ResponsiveContainer><BarChart data={data} layout="vertical" margin={{ left: 35 }}><XAxis type="number" hide /><YAxis type="category" dataKey="label" width={150} {...axisProps} tick={{ ...axisProps.tick, fontSize: 10 }} axisLine={false} /><Tooltip {...tooltipProps} formatter={(value, _name, props) => [props.payload.shap_value.toFixed(3), 'SHAP value']} /><Bar dataKey="magnitude" radius={[0, 4, 4, 0]}>{data.map((entry, index) => <Cell key={index} fill={entry.direction === 'increases_failure_risk' ? chartColors.danger : chartColors.info} />)}</Bar></BarChart></ResponsiveContainer></div>
}

// Root causes and suggestions exist only for builds submitted with a failed-test log.
function Diagnosis({ build }) {
  if (!build.root_cause_label && !build.optimization_suggestion) return null
  return <div className="mb-2 rounded-control border border-info/20 bg-info/5 p-3 text-xs">
    {build.root_cause_label && <p className="text-ink">Likely cause: {build.root_cause_label}</p>}
    {build.optimization_suggestion && <><b className="mt-1 block text-info">{build.optimization_suggestion.category}: {build.optimization_suggestion.title}</b><p className="mt-1 text-ink">{build.optimization_suggestion.suggestion}</p></>}
  </div>
}

function TimingBadge({ status }) {
  const classes = {
    normal: 'chip-success',
    bottleneck: 'chip-warn',
    stalled: 'chip-danger',
    awaiting_start: 'chip-neutral',
    missing_timestamps: 'chip-neutral',
    insufficient_history: 'chip-neutral',
  }
  return <span className={`chip ${classes[status] || classes.insufficient_history}`}>{status?.replaceAll('_', ' ') || 'unknown'}</span>
}

function JobTelemetry({ jobs, analyses, onShowAnalysis }) {
  const seconds = value => value == null ? '—' : `${Math.round(value)}s`
  return <TableShell><table className="data-table min-w-[860px]"><thead><tr><th>Job</th><th>Completed</th><th>Duration</th><th>Historical p95</th><th>Runtime assessment</th><th>Failed-job diagnostics</th></tr></thead><tbody>{jobs.map(job => {
    const analysis = analyses[job.job_run_id]
    const failed = ['failure', 'timed_out', 'cancelled', 'action_required'].includes(job.conclusion)
    return <tr key={job.job_run_id} className="align-top"><td><p className="font-medium">{job.job_name}</p><p className="text-xs text-ink-muted">{job.workflow_name || 'Workflow'}</p></td><td className="text-[13px] whitespace-nowrap text-ink-muted tabular-nums">{job.completed_at ? new Date(job.completed_at).toLocaleString() : job.status}</td><td className="stat">{seconds(job.duration_seconds)}</td><td><span className="stat">{seconds(job.baseline_p95_seconds)}</span><p className="text-xs text-ink-faint">n={job.baseline_sample_size}</p></td><td><TimingBadge status={job.bottleneck_status} /></td><td>{failed ? <>{analysis?.loading ? <span className="text-ink-muted">Loading…</span> : analysis?.error ? <span className="text-ink-muted">No captured log yet.</span> : analysis ? <div className="rounded-control border border-warn/25 bg-warn/10 p-3 text-xs"><b className="text-warn">{analysis.title}</b><p className="mt-1 text-ink">{analysis.summary}</p>{analysis.matched_signals?.length > 0 && <p className="mt-2 rounded bg-canvas/60 px-2 py-1 font-mono break-words text-ink-muted">{analysis.matched_signals[0]}</p>}</div> : <button onClick={() => onShowAnalysis(job.job_run_id)} className="btn btn-ghost btn-sm hover:border-warn/50 hover:text-warn">View diagnosis</button>}</> : <span className="text-ink-faint">—</span>}</td></tr>
  })}</tbody></table></TableShell>
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

  return <>
    <PageHeader title={pipelineId}>Pre-build failure predictions for this pipeline, newest first.</PageHeader>
    <TableShell><table className="data-table min-w-[820px]"><thead><tr><th>Timestamp</th><th>Status</th><th>Failure probability</th><th>Why</th><th>Report</th></tr></thead><tbody>
      {builds.map(build => <tr key={build.id} className="align-top">
        <td className="text-[13px] whitespace-nowrap text-ink-muted tabular-nums">{new Date(build.timestamp).toLocaleString()}</td>
        <td><StatusBadge status={build.build_status} /></td>
        <td className="stat">{build.failure_probability == null ? '—' : `${(build.failure_probability * 100).toFixed(1)}%`}</td>
        <td className="min-w-72"><Diagnosis build={build} />{build.explanation?.length ? <ExplanationChart explanation={build.explanation} /> : <span className="text-ink-faint">—</span>}</td>
        <td><button onClick={() => downloadReport(build)} className="btn btn-ghost btn-sm whitespace-nowrap">Download report</button></td>
      </tr>)}
      {!builds.length && <tr><td className="text-ink-muted" colSpan="5">No builds yet.</td></tr>}
    </tbody></table></TableShell>
    {jobs.length > 0 && <section className="mt-10"><h2 className="mb-1 text-xl text-ink"><ScrambleText text="Runtime job telemetry" /></h2><p className="mb-3 text-sm text-ink-muted">From GitHub Actions webhooks, after jobs run; not used for prediction.</p><JobTelemetry jobs={jobs} analyses={analyses} onShowAnalysis={showFailureAnalysis} /></section>}
  </>
}
