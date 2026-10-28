import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import client from '../api/client'
import { LiveDot, PageHeader, ScrambleText } from '../components/ui'
import { axisProps, chartColors, tooltipProps } from '../theme/chart'

const initialControls = {
  srcChurn: 50,
  testChurn: 10,
  filesAdded: 1,
  filesDeleted: 1,
  filesModified: 4,
  language: 'ruby',
  isPr: true,
  previousFailed: false,
}

const interventionOptions = {
  git_diff_src_churn: { label: 'Source-code churn', control: 'srcChurn', max: 2000 },
  git_diff_test_churn: { label: 'Test-code churn', control: 'testChurn', max: 2000 },
  gh_diff_files_added: { label: 'Files added', control: 'filesAdded', max: 500 },
  gh_diff_files_deleted: { label: 'Files deleted', control: 'filesDeleted', max: 500 },
  gh_diff_files_modified: { label: 'Files modified', control: 'filesModified', max: 500 },
}

// The what-if sandbox has no pipeline history, so it assumes an established
// pipeline: ten finished builds on this branch at the training failure rate
// (about 20%), nothing else running, with the toggle setting the last outcome.
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
    git_diff_src_churn: Number(controls.srcChurn),
    git_diff_test_churn: Number(controls.testChurn),
    gh_diff_files_added: Number(controls.filesAdded),
    gh_diff_files_deleted: Number(controls.filesDeleted),
    gh_diff_files_modified: Number(controls.filesModified),
    gh_lang: controls.language,
    gh_is_pr: controls.isPr,
    git_prev_commit_resolution_status: controls.previousFailed ? 'failed' : 'passed',
  }
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`
}

function RiskBadge({ probability }) {
  const value = Number(probability || 0) * 100
  const [color, bar] = value < 30 ? ['text-success', 'bg-success'] : value < 60 ? ['text-warn', 'bg-warn'] : ['text-danger', 'bg-danger']
  return <div>
    <span className={`readout text-4xl ${color}`}>{value.toFixed(1)}%</span>
    <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-raised"><div className={`sheen h-full rounded-full ${bar} transition-all duration-500`} style={{ width: `${Math.min(value, 100)}%` }} /></div>
  </div>
}

function ExplanationList({ explanation }) {
  if (!explanation?.length) return null

  return (
    <div className="mt-5 border-t border-line pt-4">
      <h3 className="label font-sans">Why the baseline risk looks this way</h3>
      <ul className="mt-3 space-y-2 text-sm">
        {explanation.slice(0, 5).map(item => (
          <li key={item.feature} className="flex justify-between gap-3">
            <span className="font-mono text-ink">{item.feature}</span>
            <span className={item.direction === 'increases_failure_risk' ? 'text-danger' : 'text-info'}>
              {item.direction === 'increases_failure_risk' ? 'increases risk' : 'decreases risk'}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
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
          <p className="readout mt-1 text-2xl">{percent(result.simulated?.failure_probability)}</p>
        </div>
      </div>

      <div className="mt-4 text-sm text-ink-muted">
        <p>
          Intervention: <span className="font-semibold text-ink">{intervention?.label || result.intervention?.field}</span>
          {' '}from <span className="stat">{Number(result.intervention?.original_value || 0)}</span>
          {' '}to <span className="stat">{Number(result.intervention?.simulated_value || 0)}</span>.
        </p>
        <p className="mt-1">Total files changed: <span className="stat">{Number(result.simulated?.derived_features?.total_files_changed || 0)}</span></p>
      </div>
      <p className="mt-4 text-xs text-ink-faint">{result.disclaimer}</p>
    </section>
  )
}

export default function WhatIfPage() {
  const [controls, setControls] = useState(initialControls)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState('')
  const [interventionField, setInterventionField] = useState('gh_diff_files_modified')
  const [interventionValue, setInterventionValue] = useState(initialControls.filesModified)
  const [simulation, setSimulation] = useState(null)
  const [simulationError, setSimulationError] = useState('')
  const [isSimulating, setIsSimulating] = useState(false)

  const features = useMemo(() => buildFeatures(controls), [controls])
  const selectedIntervention = interventionOptions[interventionField]

  useEffect(() => {
    const timer = setTimeout(async () => {
      try {
        const { data } = await client.post('/predict/whatif', { build_features: features })
        setResult(data)
        setHistory(previous => [...previous, {
          change: previous.length + 1,
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
        include_explanation: true,
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
              Source churn <span className="stat text-sm">{controls.srcChurn}</span>
              <input className="mt-2 w-full" type="range" min="0" max="2000" name="srcChurn" value={controls.srcChurn} onChange={update} />
            </label>
            <label className="label flex flex-wrap justify-between gap-x-2">
              Test churn <span className="stat text-sm">{controls.testChurn}</span>
              <input className="mt-2 w-full" type="range" min="0" max="2000" name="testChurn" value={controls.testChurn} onChange={update} />
            </label>
            <div className="grid grid-cols-3 items-end gap-3">
              {[
                ['filesAdded', 'Files added'],
                ['filesDeleted', 'Files deleted'],
                ['filesModified', 'Files modified'],
              ].map(([name, label]) => (
                <label key={name} className="label">
                  {label}
                  <input className="field" type="number" min="0" max="500" name={name} value={controls[name]} onChange={update} />
                </label>
              ))}
            </div>
            <label className="label block">
              Language
              <select className="field" name="language" value={controls.language} onChange={update}>
                {['ruby', 'java', 'python', 'javascript', 'go'].map(language => <option key={language}>{language}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2.5 text-sm text-ink"><input type="checkbox" name="isPr" checked={controls.isPr} onChange={update} className="size-4" /> Pull request build</label>
            <label className="flex items-center gap-2.5 text-sm text-ink"><input type="checkbox" name="previousFailed" checked={controls.previousFailed} onChange={update} className="size-4" /> Previous build on this branch failed</label>
            <p className="-mt-2 text-xs text-ink-faint">Assumes an established pipeline: ten finished builds on this branch at a 20% failure rate, nothing else running.</p>
          </div>
        </section>

        <section className="card tick p-5 sm:p-6">
          <p className="label flex items-center gap-2"><LiveDot />Live baseline failure probability</p>
          <div className="mt-3"><RiskBadge probability={result?.failure_probability} /></div>
          {result?.is_anomaly && <span className="chip chip-accent-2 mt-4 whitespace-normal">Unusual pre-build input profile — not a failure prediction</span>}
          {error && <p className="mt-3 text-sm text-danger" role="alert">{error}</p>}
          <div className="mt-5 h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}><CartesianGrid stroke={chartColors.grid} vertical={false} /><XAxis dataKey="change" {...axisProps} /><YAxis domain={[0, 100]} {...axisProps} axisLine={false} /><Tooltip {...tooltipProps} /><Line dataKey="probability" stroke={chartColors.accent2} strokeWidth={2} dot={false} /></LineChart>
            </ResponsiveContainer>
          </div>
          <ExplanationList explanation={simulation?.baseline?.explanation} />
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
