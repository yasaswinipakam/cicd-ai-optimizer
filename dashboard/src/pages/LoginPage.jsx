import { useState } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthContext'

export default function LoginPage() {
  const { token, login } = useAuth(); const [username, setUsername] = useState('admin'); const [password, setPassword] = useState('changeme123'); const [error, setError] = useState('')
  if (token) return <Navigate to="/" replace />
  const submit = async (event) => { event.preventDefault(); setError(''); try { await login(username, password) } catch (e) { setError(e.response?.status === 401 ? 'Invalid username or password.' : 'Unable to reach the API.') } }
  return <div className="flex min-h-screen items-center justify-center bg-slate-100"><form onSubmit={submit} className="w-full max-w-sm rounded-xl bg-white p-8 shadow"><h1 className="mb-6 text-2xl font-bold">CI/CD Optimizer</h1>{error && <p className="mb-3 rounded bg-red-50 p-3 text-sm text-red-700">{error}</p>}<label className="block text-sm">Username<input value={username} onChange={e => setUsername(e.target.value)} className="mt-1 w-full rounded border p-2" /></label><label className="mt-4 block text-sm">Password<input type="password" value={password} onChange={e => setPassword(e.target.value)} className="mt-1 w-full rounded border p-2" /></label><button className="mt-6 w-full rounded bg-sky-700 p-2 font-medium text-white hover:bg-sky-800">Sign in</button></form></div>
}
