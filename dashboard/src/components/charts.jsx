// Shared, animated chart pieces. Colors come from the theme tokens in index.css.
import { useEffect, useRef, useState } from 'react'
import { Bar, BarChart, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { featureWithValue } from './featureLabels'
import { axisProps, chartColors, tooltipProps } from '../theme/chart'

const pct = value => `${(Number(value || 0) * 100).toFixed(1)}%`

// Horizontal risk meter with the alert threshold marked; the fill animates to the new value.
export function RiskBar({ probability, threshold, label = 'the alert threshold' }) {
  const value = Math.min(Math.max(Number(probability || 0), 0), 1)
  const flagged = threshold != null && value >= threshold
  return <div>
    <div className="relative mt-3 h-2 rounded-full bg-raised">
      <div className={`sheen h-full rounded-full transition-all duration-700 ease-(--ease-out-expo) ${flagged ? 'bg-danger' : 'bg-accent'}`} style={{ width: `${value * 100}%` }} />
      {threshold != null && <div className="absolute -top-1.5 h-5 w-0.5 rounded-full bg-ink" style={{ left: `${threshold * 100}%` }} aria-hidden="true" />}
    </div>
    {threshold != null && <p className="mt-2 text-xs text-ink-muted">
      {flagged ? 'Above' : 'Below'} {label} ({pct(threshold)}, the white tick).
    </p>}
  </div>
}

// What pushed this prediction up (right, red) or down (left, blue), largest effects first.
export function ContributionChart({ explanation, height = 200 }) {
  if (!explanation?.length) return null
  const data = explanation.slice(0, 6).map(item => ({ label: featureWithValue(item.feature, item.value), value: item.shap_value }))
  return <div style={{ height }}>
    <ResponsiveContainer>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, left: 4, bottom: 4 }} barSize={12}>
        <XAxis type="number" hide domain={['dataMin', 'dataMax']} />
        <YAxis type="category" dataKey="label" width={230} {...axisProps} axisLine={false} tick={{ ...axisProps.tick, fontSize: 11 }} />
        <ReferenceLine x={0} stroke="var(--color-line-strong)" />
        <Tooltip {...tooltipProps} formatter={value => [`${value > 0 ? '+' : ''}${value.toFixed(3)} log-odds`, value > 0 ? 'Raises risk' : 'Lowers risk']} />
        <Bar dataKey="value" radius={4} animationDuration={900}>
          {data.map(item => <Cell key={item.label} fill={item.value > 0 ? chartColors.danger : chartColors.info} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
    <p className="sr-only">Red bars raise the failure risk; blue bars lower it.</p>
  </div>
}

// A tiny trend line for a table cell.
export function Sparkline({ values }) {
  if (!values?.length) return <span className="text-ink-faint">—</span>
  const data = values.map((value, index) => ({ build: index + 1, risk: +(value * 100).toFixed(1) }))
  return <div className="h-8 w-32">
    <ResponsiveContainer>
      <LineChart data={data} margin={{ top: 4, right: 4, left: 4, bottom: 4 }}>
        <YAxis hide domain={[0, 100]} />
        <Tooltip {...tooltipProps} labelFormatter={build => `Build ${build}`} formatter={value => [`${value}%`, 'Risk']} />
        <Line dataKey="risk" stroke={chartColors.accent} strokeWidth={2} dot={false} animationDuration={900} />
      </LineChart>
    </ResponsiveContainer>
  </div>
}

// Mounts its chart only once it scrolls into view, so the draw-in animation is seen.
export function ChartInView({ className, children }) {
  const ref = useRef(null)
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === 'undefined')
  useEffect(() => {
    if (seen || !ref.current) return
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setSeen(true); observer.disconnect() }
    }, { threshold: 0.3 })
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [seen])
  return <div ref={ref} className={className}>{seen && children}</div>
}
