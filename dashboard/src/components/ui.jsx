// Shared layout pieces; their look comes from the tokens and component classes in index.css.
import { useEffect, useState } from 'react'
import { animateProgress, prefersReducedMotion } from '../theme/motion'

const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&*+/<>=?'

// Decodes text left to right from random glyphs, like a display booting up.
export function ScrambleText({ text }) {
  const [progress, setProgress] = useState(0)
  useEffect(() => animateProgress(Math.min(260 + text.length * 32, 1100), setProgress), [text])
  if (prefersReducedMotion() || progress >= 1) return text
  const revealed = Math.floor(progress * text.length)
  const step = Math.floor(progress * 40)
  const shown = [...text].map((char, i) => i < revealed || char === ' ' ? char : GLYPHS[(i * 7 + step * 13) % GLYPHS.length]).join('')
  return <span aria-label={text}><span aria-hidden="true">{shown}</span></span>
}

// Counts every number inside a value (e.g. "17.3%", "87 / 9") up from zero.
export function CountUp({ value }) {
  const text = String(value)
  const [progress, setProgress] = useState(0)
  useEffect(() => animateProgress(1100, setProgress), [text])
  if (prefersReducedMotion() || progress >= 1) return text
  return text.replace(/\d+(\.\d+)?/g, match => (Number(match) * progress).toFixed(match.split('.')[1]?.length ?? 0))
}

export function Brand() {
  return <span className="flex items-center gap-2.5">
    <span className="grid size-7 place-items-center rounded-md bg-accent font-mono text-[11px] font-semibold text-white">&gt;_</span>
    <span className="font-display text-[19px] leading-none font-black text-ink">CI/CD Optimizer</span>
  </span>
}

export function PageHeader({ title, children }) {
  return <header className="mb-8">
    <h1 className="text-[30px] leading-tight break-words text-ink sm:text-[38px]">{typeof title === 'string' ? <ScrambleText text={title} /> : title}</h1>
    {children && <p className="mt-1.5 max-w-2xl animate-rise text-ink-muted [animation-delay:120ms]">{children}</p>}
  </header>
}

export function Panel({ title, subtitle, className = '', children }) {
  return <section className={`card p-5 sm:p-6 ${className}`}>
    {title && <div className="mb-4">
      <h2 className="text-lg text-ink"><ScrambleText text={title} /></h2>
      {subtitle && <p className="mt-1 text-sm text-ink-muted">{subtitle}</p>}
    </div>}
    {children}
  </section>
}

export function StatCard({ label, value, note }) {
  return <div className="card card-hover tick p-5">
    <p className="label">{label}</p>
    <p className="readout mt-2.5 text-[26px] leading-none"><CountUp value={value} /></p>
    {note && <p className="mt-2 text-xs text-ink-faint">{note}</p>}
  </div>
}

export function TableShell({ children }) {
  return <div className="card overflow-hidden"><div className="overflow-x-auto">{children}</div></div>
}

export function LiveDot() {
  return <span className="relative flex size-1.5"><span className="absolute inline-flex size-full animate-ping rounded-full bg-success opacity-70" /><span className="relative inline-flex size-1.5 rounded-full bg-success" /></span>
}

export function Loading({ children }) {
  return <p className="flex items-center gap-2.5 text-ink-muted"><span className="size-4 animate-spin rounded-full border-2 border-line-strong border-t-accent" />{children}</p>
}
