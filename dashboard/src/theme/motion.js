// Motion helpers shared by the UI components; the keyframes themselves live in index.css.
export const prefersReducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false

// One listener feeds every .card the pointer position for its hover glow.
export function trackPointerGlow() {
  document.addEventListener('pointermove', event => {
    const card = event.target instanceof Element && event.target.closest('.card')
    if (!card) return
    const rect = card.getBoundingClientRect()
    card.style.setProperty('--mx', `${event.clientX - rect.left}px`)
    card.style.setProperty('--my', `${event.clientY - rect.top}px`)
  }, { passive: true })
}

// Runs onFrame(progress) with an ease-out curve from 0 to 1 over `duration` ms; returns a cancel function.
export function animateProgress(duration, onFrame) {
  let frame
  let start
  const tick = now => {
    start ??= now
    const t = Math.min((now - start) / duration, 1)
    onFrame(1 - (1 - t) ** 3)
    if (t < 1) frame = requestAnimationFrame(tick)
  }
  frame = requestAnimationFrame(tick)
  return () => cancelAnimationFrame(frame)
}
