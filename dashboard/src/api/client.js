import axios from 'axios'

let tokenGetter = () => null
export const setTokenGetter = (getter) => { tokenGetter = getter }

const client = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000' })
client.interceptors.request.use((config) => {
  const token = tokenGetter()
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})
export default client
