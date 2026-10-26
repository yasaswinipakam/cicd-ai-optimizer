import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'
import { PageHeader, TableShell } from '../components/ui'

export default function PipelinesPage() { const [pipelines, setPipelines] = useState([]); useEffect(() => { client.get('/pipelines?limit=20&offset=0').then(r => setPipelines(r.data)) }, []); return <><PageHeader eyebrow="pipelines" title="Pipelines" /><TableShell><table className="data-table"><thead><tr><th>Pipeline</th><th>Status</th><th>Failure probability</th></tr></thead><tbody>{pipelines.map(p => <tr key={p.id}><td><Link className="link font-mono" to={`/pipelines/${encodeURIComponent(p.name)}`}>{p.name}</Link></td><td><StatusBadge status={p.latest_build_status} /></td><td className="stat">{p.latest_failure_probability == null ? '—' : `${(p.latest_failure_probability * 100).toFixed(1)}%`}</td></tr>)}{!pipelines.length && <tr><td className="text-ink-muted" colSpan="3">No pipelines yet.</td></tr>}</tbody></table></TableShell></> }
