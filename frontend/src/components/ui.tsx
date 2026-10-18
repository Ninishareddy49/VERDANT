import { motion, type HTMLMotionProps } from 'framer-motion'
import type { ReactNode } from 'react'
import { cn } from '../lib/cn'

/* ============================================================
   Primitive component kit — cards, buttons, gauges, sparklines
   ============================================================ */

type Tone = 'mint' | 'lime' | 'emerald' | 'amber' | 'danger' | 'muted'

const toneMap: Record<Tone, string> = {
  mint: 'text-mint border-mint/30 bg-mint/10',
  lime: 'text-lime border-lime/30 bg-lime/10',
  emerald: 'text-emerald border-emerald/30 bg-emerald/10',
  amber: 'text-amber border-amber/30 bg-amber/10',
  danger: 'text-danger border-danger/30 bg-danger/10',
  muted: 'text-muted border-white/10 bg-white/[0.04]',
}

/* ---------- GlassCard ---------- */
interface GlassCardProps extends HTMLMotionProps<'div'> {
  glow?: boolean
  hover?: boolean
  strong?: boolean
}
export function GlassCard({ className, glow, hover = true, strong, children, ...props }: GlassCardProps) {
  return (
    <motion.div
      className={cn(
        strong ? 'glass-strong' : 'glass',
        'rounded-2xl transition-colors duration-200',
        hover && 'hover:border-white/12',
        glow && 'glow-soft',
        className,
      )}
      whileHover={hover ? { y: -2 } : undefined}
      transition={{ type: 'spring', stiffness: 320, damping: 28 }}
      {...props}
    >
      {children}
    </motion.div>
  )
}

/* ---------- Reveal (in-view animation) ---------- */
export function Reveal({
  children,
  delay = 0,
  y = 18,
  className,
}: {
  children: ReactNode
  delay?: number
  y?: number
  className?: string
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

/* ---------- Chip / Badge ---------- */
export function Chip({
  children,
  tone = 'mint',
  className,
  dot,
}: {
  children: ReactNode
  tone?: Tone
  className?: string
  dot?: boolean
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11.5px] font-medium',
        toneMap[tone],
        className,
      )}
    >
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  )
}

/* ---------- GlowButton ---------- */
type BtnVariant = 'primary' | 'outline' | 'ghost'
interface BtnProps extends HTMLMotionProps<'button'> {
  variant?: BtnVariant
}
export function GlowButton({ variant = 'primary', className, children, ...props }: BtnProps) {
  const styles: Record<BtnVariant, string> = {
    primary: 'bg-emerald text-[#052e16] font-semibold shadow-sm hover:bg-lime',
    outline: 'border border-white/12 bg-white/[0.03] text-ink font-medium hover:bg-white/[0.07] hover:border-white/20',
    ghost: 'text-ink-dim font-medium hover:text-ink hover:bg-white/[0.06]',
  }
  return (
    <motion.button
      whileTap={{ scale: 0.98 }}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-xl px-4.5 py-2.5 text-sm transition-colors duration-150 disabled:opacity-50',
        styles[variant],
        className,
      )}
      {...props}
    >
      {children}
    </motion.button>
  )
}

/* ---------- Sparkline ---------- */
export function Sparkline({
  data,
  w = 120,
  h = 36,
  color = '#4ade80',
  fill = true,
}: {
  data: number[]
  w?: number
  h?: number
  color?: string
  fill?: boolean
}) {
  const max = Math.max(...data)
  const min = Math.min(...data)
  const span = max - min || 1
  const pad = 3
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * (w - pad * 2) + pad
    const y = h - pad - ((v - min) / span) * (h - pad * 2)
    return [x, y]
  })
  const line = pts.map((p) => p.join(',')).join(' ')
  const area = `${pad},${h - pad} ${line} ${w - pad},${h - pad}`
  const id = `sg-${color.replace('#', '')}`
  return (
    <svg width={w} height={h} className="overflow-visible">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {fill && <polygon points={area} fill={`url(#${id})`} />}
      <polyline
        points={line}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="2.6" fill={color} />
    </svg>
  )
}

/* ---------- Ring (radial gauge) ---------- */
export function Ring({
  value,
  size = 96,
  stroke = 8,
  color = '#22c55e',
  track = 'rgba(255,255,255,0.08)',
  children,
}: {
  value: number
  size?: number
  stroke?: number
  color?: string
  track?: string
  children?: ReactNode
}) {
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const off = c - (Math.min(100, Math.max(0, value)) / 100) * c
  return (
    <div className="relative inline-grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={track} strokeWidth={stroke} />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: off }}
          transition={{ duration: 1.1, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center text-center">{children}</div>
    </div>
  )
}

/* ---------- Bar (linear progress) ---------- */
export function Bar({
  value,
  color = '#22c55e',
  className,
  height = 7,
}: {
  value: number
  color?: string
  className?: string
  height?: number
}) {
  return (
    <div className={cn('w-full overflow-hidden rounded-full bg-white/8', className)} style={{ height }}>
      <motion.div
        className="h-full rounded-full"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 1, ease: 'easeOut' }}
      />
    </div>
  )
}

/* ---------- SectionTitle ---------- */
export function SectionTitle({
  eyebrow,
  title,
  sub,
  right,
  className,
}: {
  eyebrow?: string
  title: string
  sub?: string
  right?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('mb-6 flex flex-wrap items-end justify-between gap-4', className)}>
      <div>
        {eyebrow && <div className="eyebrow mb-2">{eyebrow}</div>}
        <h2 className="text-2xl font-semibold tracking-tight md:text-[26px]">{title}</h2>
        {sub && <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted">{sub}</p>}
      </div>
      {right}
    </div>
  )
}

/* ---------- LiveDot ---------- */
export function LiveDot({ label = 'LIVE' }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-[11.5px] font-medium text-lime">
      <span className="live-dot" /> {label}
    </span>
  )
}
