import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { PageHeader, Panel, ScrambleText, TableShell } from '../components/ui'
import { useModelInfo } from '../api/model'
import { ChartInView, ContributionChart } from '../components/charts'
import { axisProps, chartColors, tooltipProps } from '../theme/chart'

const OUTCOME_COLOR = { failed: chartColors.danger, passed: 'var(--color-success)' }
const OUTCOME_LABEL = { failed: 'Failed', passed: 'Passed' }

const OutcomeDot = ({ cx, cy, payload }) => <circle cx={cx} cy={cy} r={5} fill={OUTCOME_COLOR[payload.outcome] ?? 'var(--color-ink-faint)'} stroke="var(--color-surface)" strokeWidth={2} />

// Predicted risk for each build in time order, with the real outcome as the dot color.
function RiskTimeline({ builds, threshold }) {
  if (builds.length < 2) return null
  const data = [...builds].reverse().map((build, index) => ({
    build: index + 1,
    when: new Date(build.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
    risk: +(build.failure_probability * 100).toFixed(1),
    outcome: build.build_status,
  }))
  return <Panel className="mb-6" title="Risk over time" subtitle="Each dot is a build: its predicted risk, colored by what actually happened.">
    <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-muted">
      {[['Failed', chartColors.danger], ['Passed', 'var(--color-success)'], ['Outcome not reported', 'var(--color-ink-faint)']].map(([label, color]) =>
        <span key={label} className="flex items-center gap-1.5"><span className="inline-block size-2.5 rounded-full" style={{ background: color }} />{label}</span>)}
    </div>
    <ChartInView className="h-56"><ResponsiveContainer>
      <LineChart data={data} margin={{ top: 12, right: 12, left: -18, bottom: 0 }}>
        <CartesianGrid stroke={chartColors.grid} vertical={false} />
        <XAxis dataKey="build" {...axisProps} allowDecimals={false} tickFormatter={n => `#${n}`} />
        <YAxis domain={[0, 100]} unit="%" {...axisProps} axisLine={false} />
        {threshold != null && <ReferenceLine y={+(threshold * 100).toFixed(1)} stroke="var(--color-ink-muted)" strokeDasharray="4 4" label={{ value: 'alert threshold', position: 'insideTopRight', fill: 'var(--color-ink-muted)', fontSize: 10 }} />}
        <Tooltip {...tooltipProps} labelFormatter={(_n, items) => `Build #${items?.[0]?.payload.build ?? ''} · ${items?.[0]?.payload.when ?? ''}`} formatter={(value, _name, item) => [`${value}% · ${OUTCOME_LABEL[item.payload.outcome] ?? 'outcome not reported'}`, 'Predicted risk']} />
        <Line dataKey="risk" stroke={chartColors.accent} strokeWidth={2} dot={<OutcomeDot />} activeDot={{ r: 7 }} animationDuration={1200} />
      </LineChart>
    </ResponsiveContainer></ChartInView>
  </Panel>
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
  const model = useModelInfo()

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
    <RiskTimeline builds={builds.filter(build => build.failure_probability != null)} threshold={model?.decision_threshold} />
    <TableShell><table className="data-table min-w-[820px]"><thead><tr><th>Timestamp</th><th>Status</th><th>Failure probability</th><th>Why</th><th>Report</th></tr></thead><tbody>
      {builds.map(build => <tr key={build.id} className="align-top">
        <td className="text-[13px] whitespace-nowrap text-ink-muted tabular-nums">{new Date(build.timestamp).toLocaleString()}</td>
        <td><StatusBadge status={build.build_status} /></td>
        <td className="stat">{build.failure_probability == null ? '—' : `${(build.failure_probability * 100).toFixed(1)}%`}</td>
        <td className="min-w-72"><Diagnosis build={build} />{build.explanation?.length ? <ContributionChart explanation={build.explanation} height={170} /> : <span className="text-ink-faint">—</span>}</td>
        <td><button onClick={() => downloadReport(build)} className="btn btn-ghost btn-sm whitespace-nowrap">Download report</button></td>
      </tr>)}
      {!builds.length && <tr><td className="text-ink-muted" colSpan="5">No builds yet.</td></tr>}
    </tbody></table></TableShell>
    {jobs.length > 0 && <section className="mt-10"><h2 className="mb-1 text-xl text-ink"><ScrambleText text="Runtime job telemetry" /></h2><p className="mb-3 text-sm text-ink-muted">From GitHub Actions webhooks, after jobs run; not used for prediction.</p><JobTelemetry jobs={jobs} analyses={analyses} onShowAnalysis={showFailureAnalysis} /></section>}
  </>
}
