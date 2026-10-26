// Shared layout pieces; their look comes from the tokens and component classes in index.css.

export function Brand() {
  return <span className="flex items-center gap-2.5">
    <span className="grid size-8 place-items-center rounded-lg bg-linear-to-br from-accent to-accent-2 font-display text-sm font-bold text-canvas shadow-glow">&gt;_</span>
    <span className="font-display text-lg leading-none font-semibold text-ink">CI/CD<span className="text-accent">.</span>optimizer<span className="ml-0.5 inline-block h-4 w-2 translate-y-0.5 animate-blink bg-accent" /></span>
  </span>
}

export function PageHeader({ eyebrow, title, children }) {
  return <header className="mb-8">
    {eyebrow && <p className="eyebrow">{`// ${eyebrow}`}</p>}
    <h1 className="mt-2 text-3xl break-words text-ink sm:text-4xl">{title}</h1>
    {children && <p className="mt-2 max-w-2xl text-ink-muted">{children}</p>}
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
  return <div className="card card-hover group relative overflow-hidden p-5">
    <div className="pointer-events-none absolute -top-12 -right-12 size-28 rounded-full bg-accent/10 blur-2xl transition duration-500 group-hover:bg-accent/20" />
    <p className="label">{label}</p>
    <p className="stat mt-2 text-3xl">{value}</p>
  </div>
}

export function TableShell({ children }) {
  return <div className="card overflow-hidden"><div className="overflow-x-auto">{children}</div></div>
}

export function Loading({ children }) {
  return <p className="font-display text-lg text-ink-muted">{children}<span className="ml-1 inline-block h-5 w-2.5 translate-y-0.5 animate-blink bg-accent" /></p>
}
