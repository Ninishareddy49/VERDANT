/**
 * Quiet app backdrop: a single soft accent wash at the top and a faint
 * edge vignette. Static on purpose — no particles or drifting blobs, so
 * content stays the focus and idle CPU use is zero.
 */
export function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-x-0 -top-40 h-112 bg-[radial-gradient(60%_60%_at_50%_0%,rgba(74,222,128,0.08),transparent_70%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(130%_100%_at_50%_0%,transparent_60%,rgba(0,0,0,0.35)_100%)]" />
    </div>
  )
}
