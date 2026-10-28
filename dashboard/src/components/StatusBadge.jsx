export default function StatusBadge({ status }) {
  const failed = status === 'failed'
  return <span className={`chip ${failed ? 'chip-danger' : 'chip-success'}`}>{status || 'unknown'}</span>
}
