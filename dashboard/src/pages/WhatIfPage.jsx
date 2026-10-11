import { useEffect, useMemo, useState } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import client from '../api/client'

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

function buildFeatures(controls) {
  return {
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
  const color = value < 30 ? 'text-emerald-600' : value < 60 ? 'text-amber-600' : 'text-red-600'
  return <span className={`text-4xl font-bold ${color}`}>{value.toFixed(1)}%</span>
}

function ExplanationList({ explanation }) {
  if (!explanation?.length) return null

  return (
    <div className="mt-4 border-t pt-4">
      <h3 className="text-sm font-semibold text-slate-700">Why the baseline risk looks this way</h3>
      <ul className="mt-2 space-y-2 text-sm text-slate-600">
        {explanation.slice(0, 5).map(item => (
          <li key={item.feature} className="flex justify-between gap-3">
            <span>{item.feature}</span>
            <span className={item.direction === 'increases_failure_risk' ? 'text-red-600' : 'text-blue-600'}>
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
  const impactColor = delta < -0.001 ? 'text-emerald-700' : delta > 0.001 ? 'text-red-700' : 'text-slate-700'
  const intervention = interventionOptions[result.intervention?.field]

  return (
    <section className="mt-6 rounded-lg border border-indigo-200 bg-indigo-50 p-5" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-indigo-700">Optimization simulation</p>
          <h2 className="mt-1 text-lg font-bold text-slate-900">{impact}</h2>
        </div>
        <span className={`text-2xl font-bold ${impactColor}`}>{deltaPoints >= 0 ? '+' : ''}{deltaPoints.toFixed(1)} pp</span>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div className="rounded bg-white p-3 shadow-sm">
          <p className="text-xs font-medium uppercase text-slate-500">Baseline failure risk</p>
          <p className="mt-1 text-xl font-semibold text-slate-900">{percent(result.baseline?.failure_probability)}</p>
        </div>
        <div className="rounded bg-white p-3 shadow-sm">
          <p className="text-xs font-medium uppercase text-slate-500">Simulated failure risk</p>
          <p className="mt-1 text-xl font-semibold text-slate-900">{percent(result.simulated?.failure_probability)}</p>
        </div>
      </div>

      <div className="mt-4 text-sm text-slate-700">
        <p>
          Intervention: <span className="font-semibold">{intervention?.label || result.intervention?.field}</span>
          {' '}from <span className="font-semibold">{Number(result.intervention?.original_value || 0)}</span>
          {' '}to <span className="font-semibold">{Number(result.intervention?.simulated_value || 0)}</span>.
        </p>
        <p className="mt-1">Total files changed: {Number(result.simulated?.derived_features?.total_files_changed || 0)}</p>
      </div>
      <p className="mt-4 text-xs text-slate-500">{result.disclaimer}</p>
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
      <h1 className="text-2xl font-bold text-slate-900">What-If Simulator</h1>
      <p className="mt-1 text-slate-600">Explore a hypothetical build without saving anything to pipeline history.</p>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr,1.1fr]">
        <section className="rounded-lg bg-white p-5 shadow">
          <h2 className="font-semibold text-slate-900">Build inputs</h2>
          <div className="mt-4 space-y-4">
            <label className="block text-sm font-medium text-slate-700">
              Source churn: {controls.srcChurn}
              <input className="mt-1 w-full" type="range" min="0" max="2000" name="srcChurn" value={controls.srcChurn} onChange={update} />
            </label>
            <label className="block text-sm font-medium text-slate-700">
              Test churn: {controls.testChurn}
              <input className="mt-1 w-full" type="range" min="0" max="2000" name="testChurn" value={controls.testChurn} onChange={update} />
            </label>
            <div className="grid grid-cols-3 gap-3">
              {[
                ['filesAdded', 'Files added'],
                ['filesDeleted', 'Files deleted'],
                ['filesModified', 'Files modified'],
              ].map(([name, label]) => (
                <label key={name} className="text-sm font-medium text-slate-700">
                  {label}
                  <input className="mt-1 w-full rounded border border-slate-300 px-2 py-1" type="number" min="0" max="500" name={name} value={controls[name]} onChange={update} />
                </label>
              ))}
            </div>
            <label className="block text-sm font-medium text-slate-700">
              Language
              <select className="mt-1 w-full rounded border border-slate-300 p-2" name="language" value={controls.language} onChange={update}>
                {['ruby', 'java', 'python', 'javascript', 'go'].map(language => <option key={language}>{language}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700"><input type="checkbox" name="isPr" checked={controls.isPr} onChange={update} /> Pull request build</label>
            <label className="flex items-center gap-2 text-sm text-slate-700"><input type="checkbox" name="previousFailed" checked={controls.previousFailed} onChange={update} /> Previous build failed</label>
          </div>
        </section>

        <section className="rounded-lg bg-white p-5 shadow">
          <p className="text-sm font-medium text-slate-500">Live baseline failure probability</p>
          <div className="mt-2"><RiskBadge probability={result?.failure_probability} /></div>
          {result?.is_anomaly && <span className="mt-3 inline-block rounded bg-purple-100 px-2 py-1 text-sm text-purple-800">Unusual pre-build input profile — not a failure prediction</span>}
          {error && <p className="mt-3 text-sm text-red-600" role="alert">{error}</p>}
          <div className="mt-4 h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}><XAxis dataKey="change" /><YAxis domain={[0, 100]} /><Tooltip /><Line dataKey="probability" stroke="#4f46e5" dot={false} /></LineChart>
            </ResponsiveContainer>
          </div>
          <ExplanationList explanation={simulation?.baseline?.explanation} />
        </section>
      </div>

      <section className="mt-6 rounded-lg bg-white p-5 shadow">
        <h2 className="font-semibold text-slate-900">Try one intervention</h2>
        <p className="mt-1 text-sm text-slate-600">Change one actionable code-change value and compare the model's simulated risk with this baseline.</p>
        <div className="mt-4 grid gap-4 md:grid-cols-[1fr,1fr,auto] md:items-end">
          <label className="text-sm font-medium text-slate-700">
            Intervention
            <select className="mt-1 w-full rounded border border-slate-300 p-2" value={interventionField} onChange={chooseIntervention}>
              {Object.entries(interventionOptions).map(([field, option]) => <option key={field} value={field}>{option.label}</option>)}
            </select>
          </label>
          <label className="text-sm font-medium text-slate-700">
            Simulated value (0–{selectedIntervention.max})
            <input className="mt-1 w-full rounded border border-slate-300 p-2" type="number" min="0" max={selectedIntervention.max} value={interventionValue} onChange={event => { setInterventionValue(event.target.value); setSimulation(null); setSimulationError('') }} />
          </label>
          <button type="button" onClick={runSimulation} disabled={isSimulating} className="rounded bg-indigo-600 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:bg-indigo-300">
            {isSimulating ? 'Simulating…' : 'Simulate optimization'}
          </button>
        </div>
        {simulationError && <p className="mt-3 text-sm text-red-600" role="alert">{simulationError}</p>}
        <SimulationResult result={simulation} />
      </section>
    </div>
  )
}
