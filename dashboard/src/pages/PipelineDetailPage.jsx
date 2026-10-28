import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { PageHeader, ScrambleText, TableShell } from '../components/ui'
import { axisProps, chartColors, tooltipProps } from '../theme/chart'

function ExplanationChart({ explanation }) {
  if (!explanation?.length) return null
  const data = explanation.map(item => ({ ...item, magnitude: Math.abs(item.shap_value) }))
  return <div className="inset-panel mt-3 h-44 p-3 pb-8"><p className="label mb-1 text-xs">Why this prediction</p><ResponsiveContainer><BarChart data={data} layout="vertical" margin={{ left: 35 }}><XAxis type="number" hide /><YAxis type="category" dataKey="feature" width={110} {...axisProps} tick={{ ...axisProps.tick, fontSize: 10 }} axisLine={false} /><Tooltip {...tooltipProps} formatter={(value, _name, props) => [props.payload.shap_value.toFixed(3), 'SHAP value']} /><Bar dataKey="magnitude" radius={[0, 4, 4, 0]}>{data.map((entry, index) => <Cell key={index} fill={entry.direction === 'increases_failure_risk' ? chartColors.danger : chartColors.info} />)}</Bar></BarChart></ResponsiveContainer></div>
}

function RootCause({ build }) {
  if (build.root_cause_label) return <>{build.root_cause_label}</>
  if (build.build_status === 'failed' || build.predicted_status === 'failed') {
    return <span className="text-ink-muted">No failed-test log available for root-cause analysis.</span>
  }
  return <span className="text-ink-faint">—</span>
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
  if (!jobs.length) return <p className="rounded-card border border-dashed border-line-strong p-5 text-sm text-ink-muted">No GitHub Actions job telemetry has arrived for this pipeline yet.</p>
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

  return <><PageHeader title={pipelineId}>Pre-build risk scoring is separate from the runtime telemetry and post-failure evidence below.</PageHeader><section className="mb-10"><h2 className="mb-3 text-xl text-ink"><ScrambleText text="Runtime job telemetry and bottlenecks" /></h2><JobTelemetry jobs={jobs} analyses={analyses} onShowAnalysis={showFailureAnalysis} /></section><section><h2 className="mb-3 text-xl text-ink"><ScrambleText text="Pre-build predictions and completed-build history" /></h2><TableShell><table className="data-table min-w-[960px]"><thead><tr><th>Timestamp</th><th>Status</th><th>Failure probability</th><th>Input-profile anomaly</th><th>Root cause, optimization & explanation</th><th>Report</th></tr></thead><tbody>{builds.map(build => <tr key={build.id} className="align-top"><td className="text-[13px] whitespace-nowrap text-ink-muted tabular-nums">{new Date(build.timestamp).toLocaleString()}</td><td><StatusBadge status={build.build_status} /></td><td className="stat">{build.failure_probability == null ? '—' : `${(build.failure_probability * 100).toFixed(1)}%`}</td><td>{build.is_anomaly ? <><span className="chip chip-accent-2">Unusual pre-build profile</span><span className="mt-1.5 block text-xs text-ink-muted">Not a failure prediction</span></> : <span className="text-ink-muted">No unusual profile</span>}</td><td className="min-w-64"><RootCause build={build} />{build.optimization_suggestion && <div className="mt-2 rounded-control border border-info/20 bg-info/5 p-3 text-xs"><b className="text-info">{build.optimization_suggestion.category}: {build.optimization_suggestion.title}</b><p className="mt-1 text-ink">{build.optimization_suggestion.suggestion}</p></div>}<ExplanationChart explanation={build.explanation} /></td><td><button onClick={() => downloadReport(build)} className="btn btn-ghost btn-sm whitespace-nowrap">Download Report</button></td></tr>)}</tbody></table></TableShell></section></>
}
