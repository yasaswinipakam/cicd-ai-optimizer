import { BrowserRouter, Navigate, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth/AuthContext'
import LoginPage from './pages/LoginPage'
import DashboardHome from './pages/DashboardHome'
import PipelinesPage from './pages/PipelinesPage'
import PipelineDetailPage from './pages/PipelineDetailPage'
import SubmitBuildPage from './pages/SubmitBuildPage'
import WhatIfPage from './pages/WhatIfPage'
import { Brand } from './components/ui'

function ProtectedLayout() {
  const { token, user, logout } = useAuth()
  const location = useLocation()
  if (!token) return <Navigate to="/login" replace />
  const linkClass = ({ isActive }) => `shrink-0 rounded-control px-2 py-1.5 font-display text-[14px] transition sm:px-3 sm:text-[15px] sm:tracking-wide ${isActive ? 'bg-accent/10 text-accent ring-1 ring-accent/30' : 'text-ink-muted hover:bg-raised hover:text-ink'}`
  return <div className="flex min-h-screen flex-col">
    <header className="sticky top-0 z-20 border-b border-line bg-canvas/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6">
        <Brand />
        <nav className="no-scrollbar order-3 -mx-4 flex w-[calc(100%+2rem)] gap-0.5 overflow-x-auto px-4 sm:gap-1 md:order-2 md:mx-0 md:w-auto md:px-0">
          <NavLink className={linkClass} to="/" end>Dashboard</NavLink>
          <NavLink className={linkClass} to="/pipelines">Pipelines</NavLink>
          <NavLink className={linkClass} to="/submit-build">Submit Build</NavLink>
          <NavLink className={linkClass} to="/what-if">What-If</NavLink>
        </nav>
        <div className="order-2 ml-auto flex items-center gap-3 md:order-3">
          <span className="hidden font-mono text-xs text-ink-muted sm:inline">{user?.username} · {user?.role}</span>
          <button onClick={logout} className="btn btn-ghost btn-sm">Log out</button>
        </div>
      </div>
    </header>
    <main key={location.pathname} className="mx-auto w-full max-w-6xl flex-1 animate-rise px-4 py-8 sm:px-6 sm:py-10"><Routes>
      <Route path="/" element={<DashboardHome />} />
      <Route path="/pipelines" element={<PipelinesPage />} />
      <Route path="/pipelines/:pipelineId" element={<PipelineDetailPage />} />
      <Route path="/submit-build" element={<SubmitBuildPage />} />
      <Route path="/what-if" element={<WhatIfPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes></main>
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-6xl flex-wrap justify-between gap-2 px-4 py-5 font-mono text-xs text-ink-faint sm:px-6">
        <span>CI/CD Optimizer · build-failure prediction</span>
        <span>{'// signed in as '}{user?.username}</span>
      </div>
    </footer>
  </div>
}

export default function App() {
  return <AuthProvider><BrowserRouter><Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route path="/*" element={<ProtectedLayout />} />
  </Routes></BrowserRouter></AuthProvider>
}
