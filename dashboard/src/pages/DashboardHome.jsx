import { useEffect, useState } from 'react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import client from '../api/client'
import { Loading, PageHeader, Panel, ScrambleText, StatCard as Card } from '../components/ui'
import { axisProps, chartColors, tooltipProps } from '../theme/chart'
// Saved dev-test predictions; copied from reports/phase2_finance_fineract_risk_board.json.
import RISK_BOARD from '../data/fineractRiskBoard.json'

const RankedList = ({ items, empty, render }) => items.length ? <ul className="divide-y divide-line">{items.map(render)}</ul> : <p className="text-sm text-ink-muted">{empty}</p>

/* ── Finance pilot static data ── */
const COHORT = [
  { repo: 'apache/fineract',         desc: 'Core banking, loans, microfinance', suites: 1457, prevalence: '49%', span: '253 days', eligible: true,  reason: null },
  { repo: 'actualbudget/actual',     desc: 'Personal budgeting app',            suites: 2186, prevalence: '19%', span:  '90 days', eligible: false, reason: 'Needs 180-day span — recheck in 3 months' },
  { repo: 'killbill/killbill',       desc: 'Subscription billing platform',     suites:  210, prevalence: '36%', span: '390 days', eligible: false, reason: 'Needs 500+ commit suites' },
  { repo: 'firefly-iii/firefly-iii', desc: 'Personal finance tracker',          suites:   66, prevalence: '18%', span: '142 days', eligible: false, reason: 'Needs 500+ commit suites' },
]

const FOLDS = [
  { fold: 'F6',  rf: 0.579, hist: 0.627 },
  { fold: 'F7',  rf: 0.494, hist: 0.466 },
  { fold: 'F8',  rf: 0.442, hist: 0.477 },
  { fold: 'F9',  rf: 0.541, hist: 0.686 },
  { fold: 'F10', rf: 0.557, hist: 0.672 },
]

const BADGE_CLASS = {
  'payment/billing':   'border-danger/25 bg-danger/10 text-danger',
  'ledger/accounting': 'border-accent/25 bg-accent/10 text-accent',
  'auth/security':     'border-warn/25 bg-warn/10 text-warn',
  'dep/ci':            'border-line-strong bg-raised text-ink-muted',
}
const OUTCOME_CHIP = { failed: 'chip-danger', passed: 'chip-success' }

const pct = x => `${Math.round(x * 100)}%`

function FinancePilotSection() {
  const [open, setOpen] = useState(true)
  const budget = [
    ['Changes flagged', RISK_BOARD.budget_15pct],
    ['Actually failed', RISK_BOARD.failures_in_budget],
    ['Precision', pct(RISK_BOARD.precision_in_budget)],
    ['Base rate', pct(RISK_BOARD.prevalence)],
    ['Lift', `${RISK_BOARD.lift_at_15.toFixed(2)}×`],
    ['Max possible lift', `${RISK_BOARD.max_possible_lift.toFixed(2)}×`],
  ]
  return (
    <section className="mt-10">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="text-2xl text-ink"><ScrambleText text="Finance software pilot" /></h2>
          <p className="mt-1 text-sm text-ink-muted">External-validity test on 4 open-source finance repositories. Only Apache Fineract met eligibility thresholds.</p>
        </div>
        <button onClick={() => setOpen(o => !o)} className="btn btn-ghost btn-sm mt-1 shrink-0 whitespace-nowrap">
          {open ? 'Collapse ↑' : 'Expand ↓'}
        </button>
      </div>

      {open && <>
        {/* Summary cards */}
        <div className="stagger grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
          <Card label="Eligible repos" value="1 of 4" />
          <Card label="Dev-test AUC (RF, v1.2)" value="0.549" note="95% CI 0.48–0.61 · weak" />
          <Card label="Lift at 15% budget" value="1.38×" note="95% CI 1.04–1.71" />
          <Card label="Failure prevalence" value="41%" note="120 of 292 test changes failed" />
        </div>

        {/* Label-correction callout */}
        <div className="mt-4 flex animate-rise items-start gap-3 rounded-control border border-warn/30 bg-warn/8 px-4 py-3 text-sm text-warn [animation-delay:200ms]">
          <span className="mt-px shrink-0 text-base leading-none">⚠</span>
          <span><strong className="font-semibold">Corrected labels (v1.2).</strong> 56% of runs recorded as failures never executed (expired PR approvals), so the earlier below-chance result was largely a labelling artifact. On clean data the model is weak and unstable: dev-test AUC 0.45–0.55 depending on the finish buffer, with 292 test changes over 10 weeks. Same-branch history is a promising lead in online folds but not confirmed.</span>
        </div>

        {/* Cohort list + fold chart */}
        <div className="stagger mt-4 grid gap-4 lg:grid-cols-2">
          <Panel title="Cohort eligibility">
            <ul className="divide-y divide-line">
              {COHORT.map(r => (
                <li key={r.repo} className="flex items-start justify-between gap-3 py-3">
                  <div className="min-w-0">
                    <p className="font-mono text-xs break-words text-ink">{r.repo}</p>
                    <p className="mt-0.5 text-xs text-ink-muted">{r.desc}</p>
                    <p className="mt-0.5 font-mono text-xs text-ink-faint">{r.suites.toLocaleString()} suites · {r.prevalence} fail · {r.span}</p>
                    {r.reason && <p className="mt-0.5 text-xs text-ink-faint italic">{r.reason}</p>}
                  </div>
                  {r.eligible
                    ? <span className="chip chip-success shrink-0">eligible</span>
                    : <span className="chip chip-neutral shrink-0">not yet</span>}
                </li>
              ))}
            </ul>
          </Panel>

          <Panel title="Fineract online folds — AUC" subtitle="RF (v1.2, protocol) vs RF + same-branch history (v1.2, pre-specified). Random baseline = 0.5. Macro: RF 0.523, + branch history 0.586 (online folds only; not confirmed on the final test).">
            <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-muted">
              <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm" style={{background:'var(--color-ink-faint)'}} />RF (v1.2, protocol)</span>
              <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm" style={{background:'var(--color-accent)'}} />RF + same-branch history (pre-specified)</span>
            </div>
            <div className="h-44">
              <ResponsiveContainer>
                <BarChart data={FOLDS} barSize={16} barGap={2} margin={{ top: 8, right: 4, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke={chartColors.grid} vertical={false} />
                  <XAxis dataKey="fold" {...axisProps} />
                  <YAxis domain={[0, 0.7]} {...axisProps} tickFormatter={v => v.toFixed(1)} />
                  <Tooltip {...tooltipProps} formatter={v => v.toFixed(3)} />
                  <Bar dataKey="rf" name="RF" radius={[2, 2, 0, 0]} fill="var(--color-ink-faint)" />
                  <Bar dataKey="hist" name="RF + branch history" radius={[2, 2, 0, 0]} fill={chartColors.accent} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>

        {/* Top-ranked changes from the saved dev-test predictions */}
        <Panel className="mt-4" title="Top-ranked changes — Apache Fineract (v1.2 dev test, Aug 11 – Oct 14 2026)" subtitle={`The ${RISK_BOARD.top_ranked.length} highest-scoring of ${RISK_BOARD.n} test-window changes, scored by RF with protocol features (badges descriptive only), with what actually happened. Badges show the finance code areas each change touched.`}>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_16rem]">
            <ul className="divide-y divide-line">
              {RISK_BOARD.top_ranked.map(c => (
                <li key={c.sha} className="flex items-start gap-3 py-3">
                  <div className="min-w-0 flex-1">
                    <p className="font-mono text-sm text-ink">{c.sha}</p>
                    <p className="mt-0.5 text-xs text-ink-faint tabular-nums">{c.trigger_time} · {c.files} files changed</p>
                    {c.badges.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1.5">
                        {c.badges.map(b => (
                          <span key={b} className={`chip text-[11px] ${BADGE_CLASS[b] ?? 'chip-neutral'}`}>{b}</span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1.5">
                    <span className="readout text-sm text-ink-muted">{c.score.toFixed(3)}</span>
                    <span className={`chip ${OUTCOME_CHIP[c.actual] ?? 'chip-neutral'}`}>{c.actual}</span>
                  </div>
                </li>
              ))}
            </ul>
            <div className="inset-panel self-start p-4 text-sm">
              <p className="font-medium text-ink">Review budget: top 15%</p>
              <dl className="mt-3 space-y-1.5">
                {budget.map(([label, value]) => (
                  <div key={label} className="flex justify-between gap-3"><dt className="text-ink-muted">{label}</dt><dd className="readout">{value}</dd></div>
                ))}
              </dl>
              <p className="mt-3 text-xs text-ink-muted">{pct(RISK_BOARD.precision_in_budget)} of flagged changes failed against a {pct(RISK_BOARD.prevalence)} base rate (lift {RISK_BOARD.lift_at_15.toFixed(2)}×, 95% CI {RISK_BOARD.lift_ci[0].toFixed(2)}–{RISK_BOARD.lift_ci[1].toFixed(2)}): weak, not yet reliable.</p>
            </div>
          </div>
        </Panel>
      </>}
    </section>
  )
}

export default function DashboardHome() {
  const [summary, setSummary] = useState(null); const [trends, setTrends] = useState([]); const [operational, setOperational] = useState(null); const [error, setError] = useState('')
  useEffect(() => { Promise.all([client.get('/dashboard/summary'), client.get('/dashboard/failure-trends?days=30'), client.get('/dashboard/operational-summary')]).then(([s, t, o]) => { setSummary(s.data); setTrends(t.data.map(x => ({ ...x, rate: +(x.failure_rate * 100).toFixed(1) }))); setOperational(o.data) }).catch(() => setError('Unable to load dashboard data. Please sign in again.')) }, [])
  if (error) return <p className="rounded-control border border-danger/30 bg-danger/10 p-4 text-danger" role="alert">{error}</p>; if (!summary) return <Loading>Loading dashboard…</Loading>
  return <><PageHeader title="Dashboard">Build health and failure risk across monitored pipelines.</PageHeader><section className="stagger grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4"><Card label="Builds analyzed" value={summary.total_builds_analyzed} /><Card label="Failure rate" value={`${(summary.overall_failure_rate * 100).toFixed(1)}%`} /><Card label="Anomalies detected" value={summary.anomalies_detected} /><Card label="Pipelines monitored" value={summary.pipelines_monitored} /></section><Panel className="mt-6 animate-rise [animation-delay:240ms]" title="Failure rate trend">{trends.length ? <div className="h-72"><ResponsiveContainer><AreaChart data={trends} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}><defs><linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={chartColors.accent} stopOpacity={0.18} /><stop offset="100%" stopColor={chartColors.accent} stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke={chartColors.grid} vertical={false} /><XAxis dataKey="date" {...axisProps} minTickGap={24} /><YAxis unit="%" {...axisProps} axisLine={false} /><Tooltip {...tooltipProps} /><Area type="monotone" animationDuration={1400} animationEasing="ease-out" dataKey="rate" name="Failure rate" stroke={chartColors.accent} strokeWidth={2} fill="url(#trendFill)" activeDot={{ r: 4, fill: chartColors.accent, stroke: 'var(--color-canvas)', strokeWidth: 2 }} /></AreaChart></ResponsiveContainer></div> : <p className="text-sm text-ink-muted">No builds in the last 30 days.</p>}</Panel><section className="mt-10"><div className="mb-4"><h2 className="text-2xl text-ink"><ScrambleText text="Runtime bottlenecks and failure diagnostics" /></h2><p className="mt-1 text-sm text-ink-muted">Post-execution telemetry; it is not used for pre-build risk prediction.</p></div>{operational && <><div className="stagger grid gap-4 sm:grid-cols-3"><Card label="Jobs observed" value={operational.total_jobs_observed} /><Card label="Bottlenecks / stalled" value={`${operational.bottlenecks_detected} / ${operational.stalled_jobs_detected}`} /><Card label="Failed jobs with logs" value={operational.failed_jobs_with_captured_logs} /></div><div className="stagger mt-4 grid gap-4 lg:grid-cols-2"><Panel title="Slowest or recurrent bottlenecked jobs"><RankedList items={operational.top_slowest_or_bottlenecked_jobs} empty="No job telemetry yet." render={job => <li key={job.job_name} className="flex flex-col gap-0.5 py-3 sm:flex-row sm:items-baseline sm:justify-between sm:gap-3"><span className="min-w-0 font-mono text-sm break-words">{job.job_name}</span><span className="shrink-0 font-mono text-xs whitespace-nowrap text-ink-muted sm:text-right">{job.bottleneck_or_stall_count} flagged · {job.average_duration_seconds == null ? '—' : `${Math.round(job.average_duration_seconds)}s avg`}</span></li>} /></Panel><Panel title="Top structured failure categories"><RankedList items={operational.top_failure_categories} empty="No captured failed-job logs yet." render={category => <li key={category.category} className="flex justify-between gap-3 py-3"><span>{category.title}</span><span className="stat">{category.count}</span></li>} /></Panel></div></>}</section><Panel className="mt-6" title="Top root causes"><RankedList items={summary.top_root_causes} empty="No root-cause data yet." render={c => <li key={`${c.cluster_id}-${c.label}`} className="flex justify-between gap-3 py-3"><span>{c.label || `Cluster ${c.cluster_id}`}</span><span className="stat">{c.count}</span></li>} /></Panel><FinancePilotSection /></>
}
