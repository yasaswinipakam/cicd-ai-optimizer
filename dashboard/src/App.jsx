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
  const linkClass = ({ isActive }) => `relative shrink-0 px-2 py-2 text-sm font-medium transition-colors sm:px-2.5 after:absolute after:inset-x-2 after:-bottom-3 after:h-0.5 after:rounded-full after:bg-accent after:transition-transform after:duration-300 after:ease-(--ease-out-expo) ${isActive ? 'text-ink after:scale-x-100' : 'text-ink-muted after:scale-x-0 hover:text-ink hover:after:scale-x-40'}`
  return <div className="flex min-h-screen flex-col">
    <header className="sticky top-0 z-20 border-b border-line bg-canvas/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-8 gap-y-2 px-4 py-3 sm:px-6">
        <Brand />
        <nav className="no-scrollbar order-3 -mx-4 -mb-3 flex w-[calc(100%+2rem)] gap-0.5 overflow-x-auto px-4 pb-3 sm:gap-1 md:order-2 md:mx-0 md:w-auto md:px-0">
          <NavLink className={linkClass} to="/" end>Dashboard</NavLink>
          <NavLink className={linkClass} to="/pipelines">Pipelines</NavLink>
          <NavLink className={linkClass} to="/submit-build">Submit Build</NavLink>
          <NavLink className={linkClass} to="/what-if">What-If</NavLink>
        </nav>
        <div className="order-2 ml-auto flex items-center gap-3 md:order-3">
          <span className="hidden text-[13px] text-ink-muted sm:inline">{user?.username} · {user?.role}</span>
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
  </div>
}

export default function App() {
  return <AuthProvider><BrowserRouter><Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route path="/*" element={<ProtectedLayout />} />
  </Routes></BrowserRouter></AuthProvider>
}
