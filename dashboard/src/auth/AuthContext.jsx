import { createContext, useContext, useEffect, useState } from 'react'
import client, { setTokenGetter } from '../api/client'

const AuthContext = createContext(null)
export const useAuth = () => useContext(AuthContext)

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null)
  const [user, setUser] = useState(null)
  useEffect(() => setTokenGetter(() => token), [token])
  const login = async (username, password) => {
    const body = new URLSearchParams({ username, password })
    const { data } = await client.post('/auth/login', body, { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } })
    setToken(data.access_token); setUser({ username, role: data.role })
  }
  const logout = () => { setToken(null); setUser(null) }
  return <AuthContext.Provider value={{ token, user, login, logout }}>{children}</AuthContext.Provider>
}
