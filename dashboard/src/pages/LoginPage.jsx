import { useState } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'
import { Brand } from '../components/ui'

export default function LoginPage() {
  const { token, login } = useAuth(); const [username, setUsername] = useState('admin'); const [password, setPassword] = useState('changeme123'); const [error, setError] = useState('')
  if (token) return <Navigate to="/" replace />
  const submit = async (event) => { event.preventDefault(); setError(''); try { await login(username, password) } catch (e) { setError(e.response?.status === 401 ? 'Invalid username or password.' : 'Unable to reach the API.') } }
  return <div className="flex min-h-screen items-center justify-center px-4 py-12"><div className="w-full max-w-sm animate-rise"><div className="mb-8 flex justify-center"><Brand /></div><form onSubmit={submit} className="card tick p-6 sm:p-8"><h1 className="mb-1 text-3xl text-ink">Sign in</h1><p className="mb-6 text-sm text-ink-muted">Predict build failures before they run.</p>{error && <p className="mb-4 rounded-control border border-danger/30 bg-danger/10 p-3 text-sm text-danger" role="alert">{error}</p>}<label className="label block">Username<input value={username} onChange={e => setUsername(e.target.value)} className="field" autoComplete="username" /></label><label className="label mt-4 block">Password<input type="password" value={password} onChange={e => setPassword(e.target.value)} className="field" autoComplete="current-password" /></label><button className="btn btn-primary mt-6 w-full">Sign in</button></form></div></div>
}
