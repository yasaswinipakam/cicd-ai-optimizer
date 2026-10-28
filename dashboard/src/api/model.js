import { useEffect, useState } from 'react'
import client from './client'

let modelInfo
// The deployed model's name, alert threshold and held-out results, fetched once per page load.
export function useModelInfo() {
  const [info, setInfo] = useState(null)
  useEffect(() => {
    modelInfo ??= client.get('/model').then(response => response.data).catch(() => null)
    modelInfo.then(setInfo)
  }, [])
  return info
}
