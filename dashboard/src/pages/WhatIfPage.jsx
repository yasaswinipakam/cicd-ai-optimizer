import { useEffect, useMemo, useState } from 'react'
import client from '../api/client'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useModelInfo } from '../api/model'
import { ContributionChart, RiskBar } from '../components/charts'
import { LiveDot, PageHeader, ScrambleText } from '../components/ui'
import { axisProps, chartColors, tooltipProps } from '../theme/chart'

const initialControls = {
  additions: 40,
  deletions: 10,
  filesModified: 3,
  isPr: true,
  defaultBranch: false,
  isMerge: false,
  previousFailed: false,
}

const interventionOptions = {
  additions: { label: 'Lines added', control: 'additions', max: 5000 },
  deletions: { label: 'Lines deleted', control: 'deletions', max: 5000 },
  files_modified: { label: 'Files changed', control: 'filesModified', max: 500 },
}

// The what-if sandbox has no repository history, so it assumes an established
// one: ten finished runs on this branch at the training failure rate (about
// 20%), nothing else running, with the toggle setting the last outcome.
function typicalHistory(previousFailed) {
  const last = previousFailed ? 1 : 0
  return {
    hist_prev_failed: last, hist_stream_prev_failed: last, hist_prevbuild_failed: last, hist_stream_streak: last,
    hist_repo_fail_rate_5: 0.2, hist_repo_fail_rate_all: 0.2, hist_repo_ewma: 0.2,
    hist_stream_fail_rate_5: 0.2, hist_stream_fail_rate_all: 0.2, hist_stream_completed: 10,
    ctx_prevbuild_pending: 0, ctx_stream_pending: 0, ctx_repo_pending: 0, ctx_repo_started_1h: 1,
    ctx_hours_since_stream_start: 24,
  }
}

function buildFeatures(controls) {
  return {
    ...typicalHistory(controls.previousFailed),
    event: controls.isPr ? 'pull_request' : 'push',
    head_branch: controls.defaultBranch ? 'main' : 'feature',
    additions: Number(controls.additions),
    deletions: Number(controls.deletions),
    files_modified: Number(controls.filesModified),
    num_parents: controls.isMerge ? 2 : 1,
  }
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}

function RiskReadout({ probability, threshold }) {
  const flagged = threshold != null && Number(probability || 0) >= threshold
  return <div>
    <span className={`readout text-4xl ${flagged ? 'text-danger' : 'text-ink'}`}>{percent(probability)}</span>
    <RiskBar probability={probability} threshold={threshold} label="a typical repository's alert threshold" />
  </div>
}

// How the live risk moved as you edited the inputs (last 20 edits).
function RiskTrail({ history, threshold }) {
  if (history.length < 2) return <p className="mt-5 text-xs text-ink-faint">Change an input to start the risk trail.</p>
  return <div className="mt-5">
    <p className="label">Risk across your edits</p>
    <div className="mt-2 h-40">
      <ResponsiveContainer>
        <LineChart data={history} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid stroke={chartColors.grid} vertical={false} />
          <XAxis dataKey="edit" {...axisProps} />
          <YAxis domain={[0, 100]} unit="%" {...axisProps} axisLine={false} />
          {threshold != null && <ReferenceLine y={+(threshold * 100).toFixed(1)} stroke="var(--color-ink-muted)" strokeDasharray="4 4" label={{ value: 'alert threshold', position: 'insideTopRight', fill: 'var(--color-ink-muted)', fontSize: 10 }} />}
          <Tooltip {...tooltipProps} labelFormatter={edit => `Edit ${edit}`} formatter={value => [`${value}%`, 'Failure risk']} />
          <Line dataKey="probability" stroke={chartColors.accent} strokeWidth={2} dot={{ r: 3, fill: chartColors.accent, stroke: 'var(--color-surface)', strokeWidth: 2 }} activeDot={{ r: 5 }} animationDuration={600} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
}

function SimulationResult({ result }) {
  if (!result) return null

  const delta = Number(result.risk_delta || 0)
  const deltaPoints = Number(result.percentage_point_change || 0)
  const impact = delta < -0.001 ? 'Estimated risk decrease' : delta > 0.001 ? 'Estimated risk increase' : 'No material risk change'
  const impactColor = delta < -0.001 ? 'text-success' : delta > 0.001 ? 'text-danger' : 'text-ink-muted'
  const intervention = interventionOptions[result.intervention?.field]

  return (
    <section className="inset-panel mt-6 animate-rise p-5" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Optimization simulation</p>
          <h2 className="mt-1 text-xl text-ink"><ScrambleText text={impact} /></h2>
        </div>
        <span className={`readout text-3xl ${impactColor}`}>{deltaPoints >= 0 ? '+' : ''}{deltaPoints.toFixed(1)} pp</span>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div className="inset-panel p-4">
          <p className="label">Baseline failure risk</p>
          <p className="readout mt-1 text-2xl">{percent(result.baseline?.failure_probability)}</p>
        </div>
        <div className="inset-panel p-4">
          <p className="label">Simulated failure risk</p>
          <p className="readout mt-1 text-2xl">{percent(result.simulation?.failure_probability)}</p>
        </div>
      </div>

      <div className="mt-4 text-sm text-ink-muted">
        <p>
          Intervention: <span className="font-semibold text-ink">{intervention?.label || result.intervention?.field}</span>
          {' '}from <span className="stat">{Number(result.intervention?.original_value || 0)}</span>
          {' '}to <span className="stat">{Number(result.intervention?.simulated_value || 0)}</span>.
        </p>
      </div>
      <p className="mt-4 text-xs text-ink-faint">{result.disclaimer}</p>
    </section>
  )
}

export default function WhatIfPage() {
  const [controls, setControls] = useState(initialControls)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const model = useModelInfo()
  const [error, setError] = useState('')
  const [interventionField, setInterventionField] = useState('files_modified')
  const [interventionValue, setInterventionValue] = useState(initialControls.filesModified)
  const [simulation, setSimulation] = useState(null)
  const [simulationError, setSimulationError] = useState('')
  const [isSimulating, setIsSimulating] = useState(false)

  const features = useMemo(() => buildFeatures(controls), [controls])
  const selectedIntervention = interventionOptions[interventionField]

  useEffect(() => {
    const timer = setTimeout(async () => {
      try {
        const { data } = await client.post('/predict/whatif?explain=true', { build_features: features })
        setResult(data)
        setHistory(previous => [...previous, {
          edit: (previous.at(-1)?.edit ?? 0) + 1,
          probability: +(data.failure_probability * 100).toFixed(1),
        }].slice(-20))
        setError('')
      } catch {
        setError('Unable to calculate the hypothetical prediction.')
      }
    }, 400)
    return () => clearTimeout(timer)
  }, [features])

  function update(event) {
    const { name, type, checked, value } = event.target
    const next = type === 'checkbox' ? checked : value
    setControls(previous => ({ ...previous, [name]: next }))
    if (selectedIntervention.control === name) setInterventionValue(next)
    setSimulation(null)
    setSimulationError('')
  }

  function chooseIntervention(event) {
    const field = event.target.value
    setInterventionField(field)
    setInterventionValue(controls[interventionOptions[field].control])
    setSimulation(null)
    setSimulationError('')
  }

  async function runSimulation() {
    setIsSimulating(true)
    setSimulationError('')
    setSimulation(null)
    try {
      const { data } = await client.post('/predict/whatif/optimization', {
        build_features: features,
        intervention: {
          field: interventionField,
          value: Number(interventionValue),
        },
        include_explanation: false,
      })
      setSimulation(data)
    } catch (requestError) {
      setSimulationError(requestError.response?.data?.detail || 'Unable to run the optimization simulation.')
    } finally {
      setIsSimulating(false)
    }
  }

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="What-If Simulator">Explore a hypothetical build without saving anything to pipeline history.</PageHeader>

      <div className="stagger grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        <section className="card p-5 sm:p-6">
          <h2 className="text-lg text-ink"><ScrambleText text="Build inputs" /></h2>
          <div className="mt-5 space-y-5">
            <label className="label flex flex-wrap justify-between gap-x-2">
              Lines added <span className="stat text-sm">{controls.additions}</span>
              <input className="mt-2 w-full" type="range" min="0" max="2000" name="additions" value={controls.additions} onChange={update} />
            </label>
            <div className="grid grid-cols-2 items-end gap-3">
              {[
                ['deletions', 'Lines deleted', 5000],
                ['filesModified', 'Files changed', 500],
              ].map(([name, label, max]) => (
                <label key={name} className="label">
                  {label}
                  <input className="field" type="number" min="0" max={max} name={name} value={controls[name]} onChange={update} />
                </label>
              ))}
            </div>
            <label className="flex items-center gap-2.5 text-sm text-ink"><input type="checkbox" name="isPr" checked={controls.isPr} onChange={update} className="size-4" /> Pull request run</label>
            <label className="flex items-center gap-2.5 text-sm text-ink"><input type="checkbox" name="defaultBranch" checked={controls.defaultBranch} onChange={update} className="size-4" /> On the default branch (main)</label>
            <label className="flex items-center gap-2.5 text-sm text-ink"><input type="checkbox" name="isMerge" checked={controls.isMerge} onChange={update} className="size-4" /> Merge commit</label>
            <label className="flex items-center gap-2.5 text-sm text-ink"><input type="checkbox" name="previousFailed" checked={controls.previousFailed} onChange={update} className="size-4" /> Previous run on this branch failed</label>
            <p className="-mt-2 text-xs text-ink-faint">Assumes an established repository: ten finished runs on this branch at a 20% failure rate, nothing else running.</p>
          </div>
        </section>

        <section className="card tick p-5 sm:p-6">
          <p className="label flex items-center gap-2"><LiveDot />Live baseline failure probability</p>
          <div className="mt-3"><RiskReadout probability={result?.failure_probability} threshold={model?.decision_threshold} /></div>
          <RiskTrail history={history} threshold={model?.decision_threshold} />
          {error && <p className="mt-3 text-sm text-danger" role="alert">{error}</p>}
          {result?.explanation?.length > 0 && <div className="mt-5 border-t border-line pt-4"><p className="label">Why the risk looks this way</p><p className="mt-1 text-xs text-ink-faint">Red raises the risk, blue lowers it.</p><ContributionChart explanation={result.explanation} /></div>}
        </section>
      </div>

      <section className="card mt-6 p-5 sm:p-6">
        <h2 className="text-lg text-ink"><ScrambleText text="Try one intervention" /></h2>
        <p className="mt-1 text-sm text-ink-muted">Change one actionable code-change value and compare the model's simulated risk with this baseline.</p>
        <div className="mt-5 grid gap-4 md:grid-cols-[1fr_1fr_auto] md:items-end">
          <label className="label">
            Intervention
            <select className="field" value={interventionField} onChange={chooseIntervention}>
              {Object.entries(interventionOptions).map(([field, option]) => <option key={field} value={field}>{option.label}</option>)}
            </select>
          </label>
          <label className="label">
            Simulated value (0–{selectedIntervention.max})
            <input className="field" type="number" min="0" max={selectedIntervention.max} value={interventionValue} onChange={event => { setInterventionValue(event.target.value); setSimulation(null); setSimulationError('') }} />
          </label>
          <button type="button" onClick={runSimulation} disabled={isSimulating} className="btn btn-primary">
            {isSimulating ? 'Simulating…' : 'Simulate optimization'}
          </button>
        </div>
        {simulationError && <p className="mt-3 text-sm text-danger" role="alert">{simulationError}</p>}
        <SimulationResult result={simulation} />
      </section>
    </div>
  )
}
