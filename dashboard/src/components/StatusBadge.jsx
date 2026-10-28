const CHIP = { failed: 'chip-danger', passed: 'chip-success' }

export default function StatusBadge({ status }) {
  return <span className={`chip ${CHIP[status] ?? 'chip-neutral'}`}>{status || 'unknown'}</span>
}
