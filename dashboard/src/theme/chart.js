// Recharts styling that reads the design tokens defined in index.css, so charts follow the theme.
export const chartColors = {
  accent: 'var(--color-accent)',
  accent2: 'var(--color-accent-2)',
  danger: 'var(--color-danger)',
  info: 'var(--color-info)',
  grid: 'var(--color-line)',
}

export const axisProps = {
  stroke: 'var(--color-line-strong)',
  tick: { fill: 'var(--color-ink-muted)', fontSize: 11, fontFamily: 'var(--font-sans)' },
  tickLine: false,
}

export const tooltipProps = {
  contentStyle: {
    background: 'var(--color-raised)',
    border: '1px solid var(--color-line-strong)',
    borderRadius: 8,
    boxShadow: '0 12px 32px -12px rgb(0 0 0 / 0.8)',
    fontFamily: 'var(--font-sans)',
    fontSize: 12,
  },
  labelStyle: { color: 'var(--color-ink-muted)', marginBottom: 4 },
  itemStyle: { color: 'var(--color-ink)' },
  cursor: { stroke: 'var(--color-line-strong)', fill: 'rgb(148 163 184 / 0.06)' },
}
