import { useEffect, useState } from 'react'
import { CloudSun, Search } from 'lucide-react'
import { BrandMark, NAV, type View } from './Sidebar'
import { cn } from '../lib/cn'

export function Topbar({
  active,
  onNavigate,
  onHome,
}: {
  active: View
  onNavigate: (v: View) => void
  onHome: () => void
}) {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  const current = NAV.find((n) => n.id === active)!

  return (
    <header className="sticky top-0 z-30 -mx-4 mb-2 px-4 pt-4 md:-mx-8 md:px-8">
      <div className="glass flex items-center justify-between gap-4 rounded-2xl px-4 py-3 md:px-5">
        {/* mobile brand */}
        <button onClick={onHome} className="flex items-center gap-2.5 lg:hidden">
          <BrandMark size={30} />
          <span className="text-base font-bold tracking-tight">Verdant</span>
        </button>

        {/* desktop title */}
        <div className="hidden lg:block">
          <h1 className="text-[17px] font-semibold leading-none">{current.label}</h1>
          <p className="mt-1 text-[12px] text-muted">{current.hint}</p>
        </div>

        {/* command bar trigger */}
        <button
          onClick={() => window.dispatchEvent(new Event('verdant:cmdk'))}
          className="glass-soft hidden w-72 items-center gap-2 rounded-lg px-3 py-2 text-muted transition-colors hover:border-white/12 hover:text-ink-dim xl:flex"
        >
          <Search size={15} />
          <span className="text-[13px]">Search tools…</span>
          <kbd className="ml-auto rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[10.5px]">⌘K</kbd>
        </button>

        {/* right cluster */}
        <div className="flex items-center gap-3">
          <div className="glass-soft hidden items-center gap-2 rounded-lg px-3 py-1.5 sm:flex">
            <CloudSun size={15} className="text-amber" />
            <span className="font-mono text-[12.5px] text-ink-dim">22°C · Clear</span>
          </div>

          <div className="hidden text-right md:block">
            <div className="font-mono text-[13px] font-medium leading-none text-ink">
              {now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </div>
            <div className="mt-1 font-mono text-[11px] text-muted">
              {now.toLocaleDateString([], { weekday: 'short', day: '2-digit', month: 'short' })}
            </div>
          </div>

        </div>
      </div>

      {/* mobile nav pills */}
      <div className="no-scrollbar mt-3 flex gap-2 overflow-x-auto pb-1 lg:hidden">
        {NAV.map((n) => (
          <button
            key={n.id}
            onClick={() => onNavigate(n.id)}
            className={cn(
              'whitespace-nowrap rounded-lg border px-3 py-1.5 text-[12.5px] font-medium transition-colors',
              active === n.id
                ? 'border-white/12 bg-white/8 text-ink'
                : 'border-white/7 text-muted',
            )}
          >
            {n.label}
          </button>
        ))}
      </div>
    </header>
  )
}
