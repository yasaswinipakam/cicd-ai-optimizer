export default function StatusBadge({ status }) {
  const failed = status === 'failed'
  return <span className={`rounded-full px-2 py-1 text-xs font-medium ${failed ? 'bg-red-100 text-red-700' : 'bg-emerald-100 text-emerald-700'}`}>{status || 'unknown'}</span>
}
