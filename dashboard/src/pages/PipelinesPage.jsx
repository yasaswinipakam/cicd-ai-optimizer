import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { Sparkline } from '../components/charts'
import { PageHeader, TableShell } from '../components/ui'

export default function PipelinesPage() {
  const [pipelines, setPipelines] = useState([])
  useEffect(() => { client.get('/pipelines?limit=20&offset=0').then(r => setPipelines(r.data)) }, [])
  return <>
    <PageHeader title="Pipelines">Latest build status, failure risk and recent risk trend for each pipeline.</PageHeader>
    <TableShell><table className="data-table"><thead><tr><th>Pipeline</th><th>Status</th><th>Failure probability</th><th>Recent risk (last 12 builds)</th></tr></thead><tbody>
      {pipelines.map(p => <tr key={p.id}>
        <td><Link className="link font-medium" to={`/pipelines/${encodeURIComponent(p.name)}`}>{p.name}</Link></td>
        <td><StatusBadge status={p.latest_build_status} /></td>
        <td className="stat">{p.latest_failure_probability == null ? '—' : `${(p.latest_failure_probability * 100).toFixed(1)}%`}</td>
        <td><Sparkline values={p.recent_failure_probabilities} /></td>
      </tr>)}
      {!pipelines.length && <tr><td className="text-ink-muted" colSpan="4">No pipelines yet.</td></tr>}
    </tbody></table></TableShell>
  </>
}
