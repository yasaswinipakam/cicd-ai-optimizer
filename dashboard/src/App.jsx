import { BrowserRouter, Navigate, NavLink, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth/AuthContext'
import LoginPage from './pages/LoginPage'
import DashboardHome from './pages/DashboardHome'
import PipelinesPage from './pages/PipelinesPage'
import PipelineDetailPage from './pages/PipelineDetailPage'
import SubmitBuildPage from './pages/SubmitBuildPage'
import WhatIfPage from './pages/WhatIfPage'

function ProtectedLayout() {
  const { token, user, logout } = useAuth()
  if (!token) return <Navigate to="/login" replace />
  const linkClass = ({ isActive }) => `rounded px-3 py-2 text-sm ${isActive ? 'bg-sky-700 text-white' : 'text-slate-200 hover:bg-slate-700'}`
  return <div className="min-h-screen bg-slate-50 text-slate-900">
    <nav className="flex items-center gap-2 bg-slate-900 px-5 py-3 text-white">
      <span className="mr-4 font-semibold">CI/CD Optimizer</span>
      <NavLink className={linkClass} to="/">Dashboard</NavLink>
      <NavLink className={linkClass} to="/pipelines">Pipelines</NavLink>
      <NavLink className={linkClass} to="/submit-build">Submit Build</NavLink>
      <NavLink className={linkClass} to="/what-if">What-If</NavLink>
      <span className="ml-auto text-sm text-slate-300">{user?.username} · {user?.role}</span>
      <button onClick={logout} className="rounded bg-slate-700 px-3 py-2 text-sm hover:bg-slate-600">Log out</button>
    </nav>
    <main className="mx-auto max-w-6xl p-6"><Routes>
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
