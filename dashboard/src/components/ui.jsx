// Shared layout pieces; their look comes from the tokens and component classes in index.css.

export function Brand() {
  return <span className="flex items-center gap-2.5">
    <span className="grid size-7 place-items-center rounded-md bg-accent font-mono text-[11px] font-semibold text-white">&gt;_</span>
    <span className="font-display text-[19px] leading-none font-black text-ink">CI/CD Optimizer</span>
  </span>
}

export function PageHeader({ title, children }) {
  return <header className="mb-8">
    <h1 className="text-[30px] leading-tight break-words text-ink sm:text-[38px]">{title}</h1>
    {children && <p className="mt-1.5 max-w-2xl text-ink-muted">{children}</p>}
  </header>
}

export function Panel({ title, subtitle, className = '', children }) {
  return <section className={`card p-5 sm:p-6 ${className}`}>
    {title && <div className="mb-4">
      <h2 className="text-lg text-ink">{title}</h2>
      {subtitle && <p className="mt-1 text-sm text-ink-muted">{subtitle}</p>}
    </div>}
    {children}
  </section>
}

export function StatCard({ label, value }) {
  return <div className="card card-hover tick p-5">
    <p className="label">{label}</p>
    <p className="readout mt-2.5 text-[26px] leading-none">{value}</p>
  </div>
}

export function TableShell({ children }) {
  return <div className="card overflow-hidden"><div className="overflow-x-auto">{children}</div></div>
}

export function Loading({ children }) {
  return <p className="flex items-center gap-2.5 text-ink-muted"><span className="size-4 animate-spin rounded-full border-2 border-line-strong border-t-accent" />{children}</p>
}
