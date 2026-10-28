import { useEffect, useState } from 'react'
import { Bar, BarChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import client from '../api/client'
// Saved dev-test predictions; copied from reports/phase2_finance_fineract_risk_board.json.
import RISK_BOARD from '../data/fineractRiskBoard.json'

const Card = ({ label, value }) => <div className="rounded-lg bg-white p-4 shadow-sm"><p className="text-sm text-slate-500">{label}</p><p className="mt-1 text-2xl font-semibold">{value}</p></div>

const FINANCE_COHORT = [
  { repo: 'apache/fineract',         desc: 'Core banking, loans, microfinance', suites: 1457, prevalence: '49%', span: '253 days', eligible: true,  reason: null },
  { repo: 'actualbudget/actual',     desc: 'Personal budgeting app',            suites: 2186, prevalence: '19%', span:  '90 days', eligible: false, reason: 'Needs 180-day span — recheck in 3 months' },
  { repo: 'killbill/killbill',       desc: 'Subscription billing platform',     suites:  210, prevalence: '36%', span: '390 days', eligible: false, reason: 'Needs 500+ commit suites' },
  { repo: 'firefly-iii/firefly-iii', desc: 'Personal finance tracker',          suites:   66, prevalence: '18%', span: '142 days', eligible: false, reason: 'Needs 500+ commit suites' },
]

const FOLD_DATA = [
  { fold: 'F6',  rf: 0.579, hist: 0.627 },
  { fold: 'F7',  rf: 0.494, hist: 0.466 },
  { fold: 'F8',  rf: 0.442, hist: 0.477 },
  { fold: 'F9',  rf: 0.541, hist: 0.686 },
  { fold: 'F10', rf: 0.557, hist: 0.672 },
]

const badgeColors = {
  'payment/billing':   'bg-red-50 text-red-700',
  'ledger/accounting': 'bg-sky-50 text-sky-700',
  'auth/security':     'bg-amber-50 text-amber-700',
  'dep/ci':            'bg-slate-100 text-slate-600',
}

const outcomeStyle = {
  failed: 'bg-red-50 text-red-700',
  passed: 'bg-green-50 text-green-700',
}

const pct = x => `${Math.round(x * 100)}%`

function FinancePilotSection() {
  const [open, setOpen] = useState(true)
  return (
    <section className="mt-6">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="min-w-0">
          <h2 className="font-semibold">Finance software pilot</h2>
          <p className="text-sm text-slate-500">External-validity test on 4 open-source finance repositories · Apache Fineract (only eligible repo)</p>
        </div>
        <button onClick={() => setOpen(o => !o)} className="flex-shrink-0 whitespace-nowrap rounded px-3 py-1.5 text-sm bg-slate-100 hover:bg-slate-200 text-slate-600">
          {open ? 'Collapse ▲' : 'Expand ▼'}
        </button>
      </div>

      {open && <>
        {/* Key metric cards */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-4">
          <div className="rounded-lg bg-white p-4 shadow-sm">
            <p className="text-sm text-slate-500">Eligible repositories</p>
            <p className="mt-1 text-2xl font-semibold">1 <span className="text-base font-normal text-slate-400">of 4</span></p>
          </div>
          <div className="rounded-lg bg-white p-4 shadow-sm">
            <p className="text-sm text-slate-500">Dev-test AUC (RF, v1.2)</p>
            <p className="mt-1 text-2xl font-semibold text-amber-600">0.549</p>
            <p className="text-xs text-slate-400">95% CI 0.48–0.61 · weak</p>
          </div>
          <div className="rounded-lg bg-white p-4 shadow-sm">
            <p className="text-sm text-slate-500">Lift at 15% budget</p>
            <p className="mt-1 text-2xl font-semibold text-amber-600">1.38×</p>
            <p className="text-xs text-slate-400">95% CI 1.04–1.71</p>
          </div>
          <div className="rounded-lg bg-white p-4 shadow-sm">
            <p className="text-sm text-slate-500">Failure prevalence</p>
            <p className="mt-1 text-2xl font-semibold text-slate-700">41%</p>
            <p className="text-xs text-slate-400">120 of 292 test changes failed</p>
          </div>
        </div>

        {/* Label-correction callout */}
        <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          <span className="font-semibold">Corrected labels (v1.2). </span>
          56% of runs recorded as failures never executed (expired PR approvals), so the earlier below-chance result was largely a labelling artifact. On clean data the model is weak and unstable: dev-test AUC 0.45–0.55 depending on the finish buffer, with 292 test changes over 10 weeks. Same-branch history is a promising lead in online folds but not confirmed.
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          {/* Cohort table */}
          <div className="rounded-lg bg-white p-5 shadow-sm">
            <h3 className="mb-3 font-semibold">Cohort eligibility</h3>
            <ul className="divide-y text-sm">
              {FINANCE_COHORT.map(r => (
                <li key={r.repo} className="py-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <span className="font-medium font-mono text-xs break-words">{r.repo}</span>
                      <p className="text-slate-500 text-xs mt-0.5">{r.desc}</p>
                      <p className="text-xs text-slate-400 mt-0.5">{r.suites.toLocaleString()} suites · {r.prevalence} fail · {r.span}</p>
                      {!r.eligible && r.reason && <p className="text-slate-400 text-xs mt-0.5 italic">{r.reason}</p>}
                    </div>
                    {r.eligible
                      ? <span className="flex-shrink-0 whitespace-nowrap rounded bg-green-50 px-2 py-0.5 text-xs font-medium text-green-700">Eligible</span>
                      : <span className="flex-shrink-0 whitespace-nowrap rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">Not yet</span>}
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* Online folds chart */}
          <div className="rounded-lg bg-white p-5 shadow-sm">
            <h3 className="mb-1 font-semibold">Fineract online folds — AUC</h3>
            <p className="text-xs text-slate-400 mb-2">RF (v1.2, protocol) vs RF + same-branch history (v1.2, pre-specified). Random baseline = 0.5. Macro: RF 0.523, + branch history 0.586 (online folds only; not confirmed on the final test).</p>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mb-3 text-xs text-slate-500">
              <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm bg-slate-300" />RF (v1.2, protocol)</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm bg-sky-600" />RF + same-branch history (pre-specified)</span>
            </div>
            <div className="h-44">
              <ResponsiveContainer>
                <BarChart data={FOLD_DATA} barSize={14} barGap={2}>
                  <XAxis dataKey="fold" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 0.7]} tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(1)} />
                  <Tooltip formatter={(v) => v.toFixed(3)} />
                  <Bar dataKey="rf" name="RF" radius={[2, 2, 0, 0]} fill="#94a3b8" />
                  <Bar dataKey="hist" name="RF + branch history" radius={[2, 2, 0, 0]} fill="#0369a1" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Top-ranked changes from the saved dev-test predictions */}
        <div className="mt-4 rounded-lg bg-white p-5 shadow-sm">
          <h3 className="mb-1 font-semibold">Top-ranked changes — Apache Fineract (v1.2 dev test, Aug 11 – Oct 14 2026)</h3>
          <p className="text-xs text-slate-400 mb-3">
            The {RISK_BOARD.top_ranked.length} highest-scoring of {RISK_BOARD.n} test-window changes, scored by RF with protocol features (badges descriptive only), with what actually happened. Badges show the finance code areas each change touched.
          </p>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_16rem]">
            <ul className="divide-y">
              {RISK_BOARD.top_ranked.map(c => (
                <li key={c.sha} className="flex items-start gap-3 py-3">
                  <div className="min-w-0 flex-1">
                    <p className="font-mono text-sm font-medium">{c.sha}</p>
                    <p className="mt-0.5 text-xs text-slate-400">{c.trigger_time} · {c.files} files changed</p>
                    {c.badges.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {c.badges.map(b => (
                          <span key={b} className={`rounded px-1.5 py-0.5 font-mono text-xs ${badgeColors[b] ?? 'bg-slate-100 text-slate-600'}`}>{b}</span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex flex-shrink-0 flex-col items-end gap-1">
                    <span className="font-mono text-sm font-semibold text-slate-500">{c.score.toFixed(3)}</span>
                    <span className={`whitespace-nowrap rounded px-2 py-0.5 text-xs font-medium ${outcomeStyle[c.actual] ?? 'bg-slate-100 text-slate-600'}`}>{c.actual}</span>
                  </div>
                </li>
              ))}
            </ul>
            <div className="self-start rounded-lg bg-slate-50 p-4 text-sm">
              <p className="font-medium">Review budget: top 15%</p>
              <dl className="mt-2 space-y-1.5 text-slate-600">
                <div className="flex justify-between gap-3"><dt>Changes flagged</dt><dd className="font-mono">{RISK_BOARD.budget_15pct}</dd></div>
                <div className="flex justify-between gap-3"><dt>Actually failed</dt><dd className="font-mono">{RISK_BOARD.failures_in_budget}</dd></div>
                <div className="flex justify-between gap-3"><dt>Precision</dt><dd className="font-mono">{pct(RISK_BOARD.precision_in_budget)}</dd></div>
                <div className="flex justify-between gap-3"><dt>Base rate</dt><dd className="font-mono">{pct(RISK_BOARD.prevalence)}</dd></div>
                <div className="flex justify-between gap-3"><dt>Lift</dt><dd className="font-mono">{RISK_BOARD.lift_at_15.toFixed(2)}×</dd></div>
                <div className="flex justify-between gap-3"><dt>Max possible lift</dt><dd className="font-mono">{RISK_BOARD.max_possible_lift.toFixed(2)}×</dd></div>
              </dl>
              <p className="mt-3 text-xs text-slate-500">{pct(RISK_BOARD.precision_in_budget)} of flagged changes failed against a {pct(RISK_BOARD.prevalence)} base rate (lift {RISK_BOARD.lift_at_15.toFixed(2)}×, 95% CI {RISK_BOARD.lift_ci[0].toFixed(2)}–{RISK_BOARD.lift_ci[1].toFixed(2)}): weak, not yet reliable.</p>
            </div>
          </div>
        </div>
      </>}
    </section>
  )
}

export default function DashboardHome() {
  const [summary, setSummary] = useState(null); const [trends, setTrends] = useState([]); const [operational, setOperational] = useState(null); const [error, setError] = useState('')
  useEffect(() => { Promise.all([client.get('/dashboard/summary'), client.get('/dashboard/failure-trends?days=30'), client.get('/dashboard/operational-summary')]).then(([s, t, o]) => { setSummary(s.data); setTrends(t.data.map(x => ({ ...x, rate: +(x.failure_rate * 100).toFixed(1) }))); setOperational(o.data) }).catch(() => setError('Unable to load dashboard data. Please sign in again.')) }, [])
  if (error) return <p className="text-red-600">{error}</p>; if (!summary) return <p>Loading dashboard…</p>
  return <><h1 className="mb-5 text-2xl font-bold">Dashboard</h1><section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><Card label="Builds analyzed" value={summary.total_builds_analyzed} /><Card label="Failure rate" value={`${(summary.overall_failure_rate * 100).toFixed(1)}%`} /><Card label="Anomalies detected" value={summary.anomalies_detected} /><Card label="Pipelines monitored" value={summary.pipelines_monitored} /></section><section className="mt-6 rounded-lg bg-white p-5 shadow-sm"><h2 className="mb-4 font-semibold">Failure rate trend</h2><div className="h-72"><ResponsiveContainer><LineChart data={trends}><XAxis dataKey="date" /><YAxis unit="%" /><Tooltip /><Line type="monotone" dataKey="rate" name="Failure rate" stroke="#0369a1" strokeWidth={2} /></LineChart></ResponsiveContainer></div></section><section className="mt-6"><div className="mb-3"><h2 className="font-semibold">Runtime bottlenecks and failure diagnostics</h2><p className="text-sm text-slate-500">Post-execution telemetry; it is not used for pre-build risk prediction.</p></div>{operational && <><div className="grid gap-4 sm:grid-cols-3"><Card label="Jobs observed" value={operational.total_jobs_observed} /><Card label="Bottlenecks / stalled" value={`${operational.bottlenecks_detected} / ${operational.stalled_jobs_detected}`} /><Card label="Failed jobs with logs" value={operational.failed_jobs_with_captured_logs} /></div><div className="mt-4 grid gap-4 lg:grid-cols-2"><div className="rounded-lg bg-white p-5 shadow-sm"><h3 className="mb-3 font-semibold">Slowest or recurrent bottlenecked jobs</h3>{operational.top_slowest_or_bottlenecked_jobs.length ? <ul className="divide-y">{operational.top_slowest_or_bottlenecked_jobs.map(job => <li key={job.job_name} className="flex flex-col gap-0.5 py-3 sm:flex-row sm:items-baseline sm:justify-between sm:gap-3"><span className="min-w-0 break-words">{job.job_name}</span><span className="shrink-0 whitespace-nowrap text-sm text-slate-600 sm:text-right">{job.bottleneck_or_stall_count} flagged · {job.average_duration_seconds == null ? '—' : `${Math.round(job.average_duration_seconds)}s avg`}</span></li>)}</ul> : <p className="text-slate-500">No job telemetry yet.</p>}</div><div className="rounded-lg bg-white p-5 shadow-sm"><h3 className="mb-3 font-semibold">Top structured failure categories</h3>{operational.top_failure_categories.length ? <ul className="divide-y">{operational.top_failure_categories.map(category => <li key={category.category} className="flex justify-between gap-3 py-3"><span>{category.title}</span><span className="font-medium">{category.count}</span></li>)}</ul> : <p className="text-slate-500">No captured failed-job logs yet.</p>}</div></div></>}</section><section className="mt-6 rounded-lg bg-white p-5 shadow-sm"><h2 className="mb-3 font-semibold">Top root causes</h2>{summary.top_root_causes.length ? <ul className="divide-y">{summary.top_root_causes.map(c => <li key={`${c.cluster_id}-${c.label}`} className="flex justify-between py-3"><span>{c.label || `Cluster ${c.cluster_id}`}</span><span className="font-medium">{c.count}</span></li>)}</ul> : <p className="text-slate-500">No root-cause data yet.</p>}</section><FinancePilotSection /></>
}
