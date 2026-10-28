import { useEffect, useState } from 'react'
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, LabelList, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useModelInfo } from '../api/model'
import { ChartInView } from '../components/charts'
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
  const budget = [
    ['Changes flagged', RISK_BOARD.budget_15pct],
    ['Actually failed', RISK_BOARD.failures_in_budget],
    ['Precision', pct(RISK_BOARD.precision_in_budget)],
    ['Base rate', pct(RISK_BOARD.prevalence)],
  ]
  return (
    <section className="mt-10">
      <div className="mb-4">
        <h2 className="text-2xl text-ink"><ScrambleText text="Finance software pilot" /></h2>
        <p className="mt-1 text-sm text-ink-muted">External-validity test on 4 open-source finance repositories. Only Apache Fineract met eligibility thresholds.</p>
      </div>

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
              <p className="mt-3 text-xs text-ink-muted">Weak: not yet reliable enough to act on.</p>
            </div>
          </div>
        </Panel>
    </section>
  )
}

// Within-repository AUC from the Phase-1 reports (0.5 = random ranking).
// Travis: walk-forward folds 7-10; GitHub Actions: pre-registered external test, folds 6-10.
const MODEL_RESULTS = [
  { set: 'All builds (Travis)', tierC: 0.628, tierE: 0.698 },
  { set: 'New breakages (Travis)', tierC: 0.556, tierE: 0.622 },
  { set: 'GitHub Actions (external)', tierC: 0.715, tierE: 0.770 },
]

function RiskDistribution({ distribution, threshold }) {
  const data = distribution.bins.map(bin => ({ mid: Math.round((bin.start + bin.end) * 50), range: `${Math.round(bin.start * 100)}–${Math.round(bin.end * 100)}%`, count: bin.count }))
  const subtitle = distribution.flagged == null ? 'Failure probability of every scored build.' : `${distribution.flagged} of ${distribution.total} scored builds are at or above the current alert threshold.`
  return <Panel title="Predicted risk distribution" subtitle={subtitle}>
    {distribution.total ? <ChartInView className="h-64"><ResponsiveContainer>
      <BarChart data={data} margin={{ top: 16, right: 8, left: -18, bottom: 0 }} barCategoryGap={2}>
        <CartesianGrid stroke={chartColors.grid} vertical={false} />
        <XAxis type="number" dataKey="mid" domain={[0, 100]} ticks={[0, 20, 40, 60, 80, 100]} unit="%" {...axisProps} />
        <YAxis allowDecimals={false} {...axisProps} axisLine={false} />
        <Tooltip {...tooltipProps} labelFormatter={(_mid, items) => `Risk ${items?.[0]?.payload.range ?? ''}`} formatter={value => [value, 'Builds']} />
        {threshold != null && <ReferenceLine x={threshold * 100} stroke="var(--color-ink-muted)" strokeDasharray="4 4" label={{ value: 'alert threshold', position: 'top', fill: 'var(--color-ink-muted)', fontSize: 10 }} />}
        <Bar dataKey="count" fill={chartColors.accent} radius={[4, 4, 0, 0]} animationDuration={1100} />
      </BarChart>
    </ResponsiveContainer></ChartInView> : <p className="text-sm text-ink-muted">No scored builds yet.</p>}
  </Panel>
}

function ModelPerformance({ model }) {
  const holdout = model?.holdout_metrics
  return <Panel title="How good is the model?" subtitle="Within-repository ranking quality (AUC); the axis starts at 0.5, which is random.">
    <ChartInView className="h-64"><ResponsiveContainer>
      <BarChart data={MODEL_RESULTS} layout="vertical" margin={{ top: 0, right: 36, left: 8, bottom: 0 }} barGap={2} barSize={12}>
        <CartesianGrid stroke={chartColors.grid} horizontal={false} />
        <XAxis type="number" domain={[0.5, 0.8]} ticks={[0.5, 0.6, 0.7, 0.8]} {...axisProps} />
        <YAxis type="category" dataKey="set" width={150} {...axisProps} axisLine={false} tick={{ ...axisProps.tick, fontSize: 11 }} />
        <Tooltip {...tooltipProps} formatter={value => value.toFixed(3)} />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} formatter={value => <span style={{ color: 'var(--color-ink-muted)' }}>{value}</span>} />
        <Bar dataKey="tierC" name="Previous model (Tier C)" fill="var(--color-ink-faint)" radius={[0, 4, 4, 0]} animationDuration={1100} />
        <Bar dataKey="tierE" name="Deployed model (Tier E)" fill={chartColors.accent} radius={[0, 4, 4, 0]} animationDuration={1300}>
          <LabelList dataKey="tierE" position="right" formatter={value => value.toFixed(2)} style={{ fill: 'var(--color-ink)', fontSize: 11 }} />
        </Bar>
      </BarChart>
    </ResponsiveContainer></ChartInView>
    {holdout && <dl className="mt-4 grid grid-cols-3 gap-3 border-t border-line pt-4 text-sm">
      {[['Flags', holdout.alert_rate, 'of builds'], ['Precision', holdout.precision_at_threshold, 'of flags fail'], ['Recall', holdout.recall_at_threshold, 'of failures caught']].map(([label, value, note]) =>
        <div key={label}><dt className="label">{label}</dt><dd className="readout mt-1 text-lg">{Math.round(value * 100)}%</dd><dd className="text-xs text-ink-faint">{note}</dd></div>)}
    </dl>}
    {holdout && <p className="mt-2 text-xs text-ink-faint">Deployed model at its alert threshold, on its held-out data (most recent 10% of Travis builds).</p>}
  </Panel>
}

const TREND_DAYS = 90
const formatDay = ms => new Date(ms).toLocaleDateString([], { month: 'short', day: 'numeric', timeZone: 'UTC' })

// Daily average predicted risk (area) with the actual failure rate where outcomes were reported (line).
const MIN_DAYS_FOR_DAILY = 3
const toPercent = value => (value == null ? null : +(value * 100).toFixed(1))

async function loadTrend() {
  const params = { days: TREND_DAYS }
  const daily = (await client.get('/dashboard/failure-trends', { params })).data
  if (daily.length >= MIN_DAYS_FOR_DAILY) {
    return { mode: 'day', points: daily.map(d => { const x = Date.parse(`${d.date}T00:00:00Z`); return { x, when: formatDay(x), risk: toPercent(d.average_predicted_risk), rate: toPercent(d.failure_rate), scored: d.scored_builds, reported: d.total_builds } }) }
  }
  // Too few days for a daily line: plot each build in order instead.
  const builds = (await client.get('/dashboard/failure-trends', { params: { ...params, granularity: 'build' } })).data
  return { mode: 'build', points: builds.map((b, i) => ({ x: i + 1, when: new Date(b.date).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }), risk: toPercent(b.average_predicted_risk), rate: toPercent(b.failure_rate), scored: 1, reported: b.total_builds })) }
}

// Predicted risk (area) with the actual failure rate from reported outcomes (line).
function FailureTrend({ threshold }) {
  const [trend, setTrend] = useState(null)
  useEffect(() => { loadTrend().then(setTrend).catch(() => setTrend({ mode: 'day', points: [] })) }, [])
  if (!trend) return null
  const byBuild = trend.mode === 'build'
  const subtitle = byBuild
    ? `Fewer than ${MIN_DAYS_FOR_DAILY} days of builds so far, so each build is shown in order: its predicted risk, and the failure rate of reported outcomes up to that build.`
    : `Last ${TREND_DAYS} days: average predicted risk per day, and the actual failure rate on days with reported outcomes.`
  const names = byBuild ? { risk: 'Predicted risk', rate: 'Failure rate so far' } : { risk: 'Average predicted risk', rate: 'Actual failure rate' }
  return <Panel className="mt-6 animate-rise [animation-delay:240ms]" title="Failure rate trend" subtitle={subtitle}>
    {trend.points.length ? <ChartInView className="h-72"><ResponsiveContainer>
      <ComposedChart data={trend.points} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
        <defs><linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={chartColors.accent} stopOpacity={0.3} /><stop offset="100%" stopColor={chartColors.accent} stopOpacity={0} /></linearGradient></defs>
        <CartesianGrid stroke={chartColors.grid} vertical={false} />
        {byBuild
          ? <XAxis type="number" dataKey="x" domain={['dataMin', 'dataMax']} allowDecimals={false} tickFormatter={n => `#${n}`} {...axisProps} />
          : <XAxis type="number" scale="time" dataKey="x" domain={['dataMin', 'dataMax']} tickFormatter={formatDay} {...axisProps} minTickGap={32} />}
        <YAxis domain={[0, 100]} unit="%" {...axisProps} axisLine={false} />
        {threshold != null && <ReferenceLine y={+(threshold * 100).toFixed(1)} stroke="var(--color-ink-muted)" strokeDasharray="4 4" label={{ value: 'alert threshold', position: 'insideTopRight', fill: 'var(--color-ink-muted)', fontSize: 10 }} />}
        <Tooltip {...tooltipProps} labelFormatter={(_x, items) => items?.[0]?.payload.when ?? ''} formatter={(value, name, item) => [value == null ? '—' : `${value}%`, name === 'risk' ? (byBuild ? names.risk : `${names.risk} (${item.payload.scored} builds)`) : `${names.rate} (${item.payload.reported} reported)`]} />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} formatter={name => <span style={{ color: 'var(--color-ink-muted)' }}>{names[name]}</span>} />
        <Area type="monotone" dataKey="risk" stroke={chartColors.accent} strokeWidth={2} fill="url(#trendFill)" connectNulls animationDuration={1800} animationEasing="ease-out" activeDot={{ r: 5, fill: chartColors.accent, stroke: 'var(--color-canvas)', strokeWidth: 2 }} dot={{ r: 3, fill: chartColors.accent, stroke: 'var(--color-canvas)', strokeWidth: 2 }} />
        <Line type="monotone" dataKey="rate" stroke={chartColors.accent2} strokeWidth={2} connectNulls animationBegin={300} animationDuration={1800} animationEasing="ease-out" dot={{ r: 4, fill: chartColors.accent2, stroke: 'var(--color-canvas)', strokeWidth: 2 }} activeDot={{ r: 6 }} />
      </ComposedChart>
    </ResponsiveContainer></ChartInView> : <p className="text-sm text-ink-muted">No scored builds in the last {TREND_DAYS} days.</p>}
  </Panel>
}

// Runtime telemetry arrives from GitHub Actions webhooks after jobs run; it is not used for prediction.
function RuntimeSection({ operational }) {
  return <section className="mt-10">
    <div className="mb-4"><h2 className="text-2xl text-ink"><ScrambleText text="Runtime bottlenecks and failure diagnostics" /></h2><p className="mt-1 text-sm text-ink-muted">From GitHub Actions webhooks, after jobs run; not used for prediction.</p></div>
    <div className="stagger grid gap-4 sm:grid-cols-3"><Card label="Jobs observed" value={operational.total_jobs_observed} /><Card label="Bottlenecks / stalled" value={`${operational.bottlenecks_detected} / ${operational.stalled_jobs_detected}`} /><Card label="Failed jobs with logs" value={operational.failed_jobs_with_captured_logs} /></div>
    <div className="stagger mt-4 grid gap-4 lg:grid-cols-2">
      <Panel title="Slowest or recurrent bottlenecked jobs"><RankedList items={operational.top_slowest_or_bottlenecked_jobs} empty="No bottlenecked jobs." render={job => <li key={job.job_name} className="flex flex-col gap-0.5 py-3 sm:flex-row sm:items-baseline sm:justify-between sm:gap-3"><span className="min-w-0 font-mono text-sm break-words">{job.job_name}</span><span className="shrink-0 font-mono text-xs whitespace-nowrap text-ink-muted sm:text-right">{job.bottleneck_or_stall_count} flagged · {job.average_duration_seconds == null ? '—' : `${Math.round(job.average_duration_seconds)}s avg`}</span></li>} /></Panel>
      <Panel title="Top structured failure categories"><RankedList items={operational.top_failure_categories} empty="No captured failed-job logs." render={category => <li key={category.category} className="flex justify-between gap-3 py-3"><span>{category.title}</span><span className="stat">{category.count}</span></li>} /></Panel>
    </div>
  </section>
}

export default function DashboardHome() {
  const [summary, setSummary] = useState(null); const [operational, setOperational] = useState(null); const [distribution, setDistribution] = useState(null); const [error, setError] = useState('')
  const model = useModelInfo()
  // Count flags against the current model's threshold once it is known.
  useEffect(() => { client.get('/dashboard/risk-distribution', { params: { threshold: model?.decision_threshold } }).then(r => setDistribution(r.data)).catch(() => setDistribution(null)) }, [model])
  useEffect(() => { Promise.all([client.get('/dashboard/summary'), client.get('/dashboard/operational-summary')]).then(([s, o]) => { setSummary(s.data); setOperational(o.data) }).catch(() => setError('Unable to reach the API. Is it running?')) }, [])
  if (error) return <p className="rounded-control border border-danger/30 bg-danger/10 p-4 text-danger" role="alert">{error}</p>
  if (!summary) return <Loading>Loading dashboard…</Loading>
  return <>
    <PageHeader title="Dashboard">Build health and failure risk across monitored pipelines.</PageHeader>
    <section className="stagger grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-3"><Card label="Builds analyzed" value={summary.total_builds_analyzed} /><Card label="Failure rate" value={summary.builds_with_outcome ? `${(summary.overall_failure_rate * 100).toFixed(1)}%` : '—'} note={`${summary.builds_with_outcome} builds with a reported outcome`} /><Card label="Pipelines monitored" value={summary.pipelines_monitored} /></section>
    <FailureTrend threshold={model?.decision_threshold} />
    <div className="stagger mt-6 grid gap-4 lg:grid-cols-2">{distribution && <RiskDistribution distribution={distribution} threshold={model?.decision_threshold} />}<ModelPerformance model={model} /></div>
    {summary.top_root_causes.length > 0 && <Panel className="mt-6" title="Top root causes" subtitle="From failed-test logs attached to failed builds."><RankedList items={summary.top_root_causes} empty="" render={c => <li key={`${c.cluster_id}-${c.label}`} className="flex justify-between gap-3 py-3"><span>{c.label || `Cluster ${c.cluster_id}`}</span><span className="stat">{c.count}</span></li>} /></Panel>}
    {operational?.total_jobs_observed > 0 && <RuntimeSection operational={operational} />}
    <FinancePilotSection />
  </>
}
