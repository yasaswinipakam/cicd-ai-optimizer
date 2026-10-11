import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import client from '../api/client'
import StatusBadge from '../components/StatusBadge'

export default function PipelinesPage() { const [pipelines, setPipelines] = useState([]); useEffect(() => { client.get('/pipelines?limit=20&offset=0').then(r => setPipelines(r.data)) }, []); return <><h1 className="mb-5 text-2xl font-bold">Pipelines</h1><div className="overflow-hidden rounded-lg bg-white shadow-sm"><table className="w-full text-left text-sm"><thead className="bg-slate-100 text-slate-600"><tr><th className="p-3">Pipeline</th><th>Status</th><th>Failure probability</th></tr></thead><tbody>{pipelines.map(p => <tr key={p.id} className="border-t"><td className="p-3"><Link className="text-sky-700 hover:underline" to={`/pipelines/${encodeURIComponent(p.name)}`}>{p.name}</Link></td><td><StatusBadge status={p.latest_build_status} /></td><td>{p.latest_failure_probability == null ? '—' : `${(p.latest_failure_probability * 100).toFixed(1)}%`}</td></tr>)}{!pipelines.length && <tr><td className="p-4 text-slate-500" colSpan="3">No pipelines yet.</td></tr>}</tbody></table></div></> }
