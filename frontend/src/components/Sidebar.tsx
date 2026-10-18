import { motion } from 'framer-motion'
import {
  Beaker,
  Bot,
  CalendarRange,
  Gauge,
  LayoutDashboard,
  LayoutGrid,
  Leaf,
  ScanLine,
  Stethoscope,
  type LucideIcon,
} from 'lucide-react'
import { cn } from '../lib/cn'

export type View =
  | 'overview'
  | 'doctor'
  | 'scanner'
  | 'simulator'
  | 'vpd'
  | 'growplan'
  | 'nutrients'
  | 'advisor'

export interface NavItem {
  id: View
  label: string
  icon: LucideIcon
  hint: string
}

export const NAV: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, hint: 'Command center' },
  { id: 'doctor', label: 'Plant Doctor', icon: Stethoscope, hint: 'Disease diagnosis' },
  { id: 'scanner', label: 'Species Scanner', icon: ScanLine, hint: 'Identify plants' },
  { id: 'simulator', label: 'Grow Simulator', icon: Gauge, hint: 'Digital twin' },
  { id: 'vpd', label: 'VPD Matrix', icon: LayoutGrid, hint: 'Climate console' },
  { id: 'growplan', label: 'Grow Plan', icon: CalendarRange, hint: 'AI scheduler' },
  { id: 'nutrients', label: 'Nutrient Lab', icon: Beaker, hint: 'Dosing solver' },
  { id: 'advisor', label: 'AI Advisor', icon: Bot, hint: 'Ask anything' },
]

const GROUPS: { label: string; ids: View[] }[] = [
  { label: 'Diagnose', ids: ['doctor', 'scanner'] },
  { label: 'Grow Lab', ids: ['simulator', 'vpd', 'growplan', 'nutrients'] },
  { label: 'Assist', ids: ['advisor'] },
]

const byId = (id: View) => NAV.find((n) => n.id === id)!

/** Flat vector brand mark — replaces the glowing raster logo in the app shell. */
export function BrandMark({ size = 36 }: { size?: number }) {
  return (
    <span
      className="grid shrink-0 place-items-center rounded-xl bg-emerald text-[#052e16] shadow-sm"
      style={{ width: size, height: size }}
    >
      <Leaf size={size * 0.52} strokeWidth={2.2} />
    </span>
  )
}

export function Sidebar({ active, onNavigate, onHome }: { active: View; onNavigate: (v: View) => void; onHome: () => void }) {
  return (
    <aside className="glass-strong fixed inset-y-0 left-0 z-40 hidden w-[260px] flex-col border-y-0 border-l-0 px-4 py-6 lg:flex">
      {/* brand */}
      <button onClick={onHome} className="mb-8 flex shrink-0 items-center gap-3 rounded-xl px-2 text-left">
        <BrandMark />
        <div>
          <div className="text-[17px] font-bold leading-none tracking-tight">Verdant</div>
          <div className="mt-1 text-[11.5px] text-muted">Urban farming intelligence</div>
        </div>
      </button>

      {/* nav */}
      <nav className="no-scrollbar flex flex-1 flex-col gap-0.5 overflow-y-auto">
        <NavButton item={byId('overview')} active={active} onNavigate={onNavigate} />

        {GROUPS.map((g) => (
          <div key={g.label} className="mt-5">
            <div className="mb-1.5 px-3 text-[11px] font-semibold uppercase tracking-wider text-faint">{g.label}</div>
            {g.ids.map((id) => (
              <NavButton key={id} item={byId(id)} active={active} onNavigate={onNavigate} />
            ))}
          </div>
        ))}
      </nav>

      {/* footer status card */}
      <div className="mt-4 shrink-0 rounded-xl border border-white/7 bg-white/[0.025] p-3.5">
        <div className="flex items-center gap-2 text-[12.5px] font-medium text-ink">
          <span className="live-dot" /> All systems operational
        </div>
        <p className="mt-1.5 text-[12px] leading-relaxed text-muted">
          8 tools · <kbd className="rounded border border-white/10 bg-white/5 px-1 text-[10.5px] text-ink-dim">⌘K</kbd> to search
        </p>
        <div className="mt-2.5 flex items-center gap-1.5 text-[11.5px] text-muted">
          <Leaf size={12} className="text-emerald" /> Carbon-aware operations
        </div>
      </div>
    </aside>
  )
}

function NavButton({ item, active, onNavigate }: { item: NavItem; active: View; onNavigate: (v: View) => void }) {
  const isActive = active === item.id
  const Icon = item.icon
  return (
    <button
      onClick={() => onNavigate(item.id)}
      className={cn(
        'group relative flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left transition-colors duration-150',
        isActive ? 'text-ink' : 'text-muted hover:bg-white/[0.04] hover:text-ink',
      )}
    >
      {isActive && (
        <motion.div
          layoutId="nav-active"
          className="absolute inset-0 rounded-xl border border-white/8 bg-white/[0.06]"
          transition={{ type: 'spring', stiffness: 420, damping: 34 }}
        />
      )}
      <Icon size={18} strokeWidth={1.9} className={cn('relative shrink-0', isActive ? 'text-lime' : 'text-faint group-hover:text-ink-dim')} />
      <span className="relative min-w-0">
        <span className="block truncate text-[13.5px] font-medium leading-tight">{item.label}</span>
        <span className="block truncate text-[11.5px] leading-tight text-faint">{item.hint}</span>
      </span>
    </button>
  )
}
