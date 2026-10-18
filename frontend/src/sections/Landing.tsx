import { motion } from 'framer-motion'
import { ArrowRight, Bot, ScanLine, Stethoscope, Droplets, Zap, Leaf, Cpu } from 'lucide-react'
import { Reveal } from '../components/ui'
import { BrandMark } from '../components/Sidebar'

const HOW: {
  icon: typeof Stethoscope
  color: string
  eyebrow: string
  title: string
  desc: string
  tag: string
}[] = [
  {
    icon: Stethoscope,
    color: '#22c55e',
    eyebrow: 'Vision Transformer · 13 classes',
    title: 'Plant Doctor',
    desc: 'Photograph a crop leaf and get an instant diagnosis — condition, confidence, severity, and a treatment and prevention plan. Covers common diseases of corn, potato, rice and wheat.',
    tag: 'Upload a leaf → diagnosis',
  },
  {
    icon: ScanLine,
    color: '#4ade80',
    eyebrow: 'Vision Transformer · 52 species',
    title: 'Species Scanner',
    desc: 'Identify a plant from a photo across 52 medicinal and crop species, then get a tailored grow profile with ideal water, light and days-to-harvest.',
    tag: 'Photo → species + grow profile',
  },
  {
    icon: Bot,
    color: '#34d399',
    eyebrow: 'Gemma 3 · Agronomy knowledge base',
    title: 'AI Advisor',
    desc: 'Ask growing questions in plain English. Semantic search over a curated agronomy knowledge base grounds every answer, with Gemma 3 adding a concise explanation.',
    tag: 'Question → grounded answer',
  },
]

const STATS = [
  { value: '95%', label: 'Less water than soil farming', icon: Droplets },
  { value: '98%', label: 'Reported disease-model accuracy', icon: Zap },
  { value: '52', label: 'Plant species identified', icon: Leaf },
  { value: 'CPU', label: 'Runs locally, no GPU needed', icon: Cpu },
]

export function Landing({ onLaunch }: { onLaunch: () => void }) {
  return (
    <motion.div
      className="relative"
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
    >
      {/* ═══════════════════════ HERO ═══════════════════════ */}
      <section className="relative flex min-h-screen flex-col">
        {/* top bar */}
        <nav className="relative z-10 mx-auto flex w-full max-w-[1180px] items-center justify-between px-6 py-5 md:px-10">
          <Reveal>
            <div className="flex items-center gap-3">
              <BrandMark size={34} />
              <div>
                <div className="text-[17px] font-bold leading-none tracking-tight">Verdant</div>
                <div className="mt-1 text-[11.5px] text-muted">Urban farming intelligence</div>
              </div>
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <button
              onClick={onLaunch}
              className="hidden items-center gap-2 rounded-xl border border-white/12 bg-white/3 px-4 py-2 text-[13.5px] font-medium text-ink transition-colors hover:bg-white/7 sm:flex"
            >
              Launch Platform <ArrowRight size={14} />
            </button>
          </Reveal>
        </nav>

        {/* center hero content */}
        <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-6 pb-16 pt-6 text-center md:px-10">
          <Reveal>
            <span className="mb-7 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/3 px-3.5 py-1.5 text-[12.5px] font-medium text-ink-dim">
              <span className="live-dot" />
              AI-powered decision support for vertical farms
            </span>
          </Reveal>

          <Reveal delay={0.06}>
            <h1 className="max-w-4xl pb-2 text-[3rem] font-bold leading-[1.1] tracking-[-0.03em] sm:text-[4.2rem] lg:text-[5.2rem]">
              Grow smarter,
              <br />
              <span className="text-grad">in the city.</span>
            </h1>
          </Reveal>

          <Reveal delay={0.12}>
            <p className="mt-8 max-w-2xl text-[17px] leading-relaxed text-ink-dim">
              Verdant brings machine learning to urban growers — diagnose plant disease, identify species, plan
              climate and nutrients, and get grounded agronomic answers, all from one platform that runs on an
              ordinary laptop.
            </p>
          </Reveal>

          {/* CTAs */}
          <Reveal delay={0.18}>
            <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
              <motion.button
                onClick={onLaunch}
                className="inline-flex items-center gap-2.5 rounded-xl bg-emerald px-6 py-3.5 text-[15px] font-semibold text-[#052e16] shadow-[0_8px_24px_-10px_rgba(34,197,94,0.6)] transition-colors hover:bg-lime"
                whileTap={{ scale: 0.98 }}
              >
                Enter Verdant
                <ArrowRight size={18} />
              </motion.button>
              {[
                { icon: Stethoscope, label: 'Plant Doctor' },
                { icon: ScanLine, label: 'Species Scanner' },
                { icon: Bot, label: 'AI Advisor' },
              ].map((f) => (
                <button
                  key={f.label}
                  onClick={onLaunch}
                  className="hidden items-center gap-2 rounded-xl border border-white/10 bg-white/3 px-4 py-3.5 text-[14px] font-medium text-ink-dim transition-colors hover:border-white/20 hover:text-ink md:inline-flex"
                >
                  <f.icon size={16} className="text-lime" />
                  {f.label}
                </button>
              ))}
            </div>
          </Reveal>

          {/* stats strip */}
          <Reveal delay={0.24}>
            <div className="mt-16 grid w-full max-w-4xl grid-cols-2 overflow-hidden rounded-2xl border border-white/8 bg-white/2 md:grid-cols-4">
              {STATS.map((s, i) => (
                <div
                  key={s.label}
                  className={`px-5 py-5 text-left ${i % 2 === 1 ? 'border-l border-white/8' : ''} ${i === 2 ? 'md:border-l md:border-white/8' : ''} ${i >= 2 ? 'border-t border-white/8 md:border-t-0' : ''}`}
                >
                  <s.icon size={16} className="mb-3 text-lime" />
                  <div className="font-mono text-[28px] font-bold leading-none tracking-tight">{s.value}</div>
                  <div className="mt-1.5 text-[12.5px] text-muted">{s.label}</div>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══════════════════════ HOW IT WORKS ═══════════════════════ */}
      <section className="mx-auto max-w-[1100px] px-6 pb-20 md:px-10">
        <Reveal>
          <div className="mb-10 text-center">
            <span className="eyebrow mb-3 block">How it works</span>
            <h2 className="text-[2rem] font-bold tracking-tight">Three AI tools. One platform.</h2>
            <p className="mx-auto mt-3 max-w-xl text-[15px] text-muted">
              Every model runs on the CPU behind a single API, so the platform works offline-first and costs nothing to run.
            </p>
          </div>
        </Reveal>

        <div className="grid gap-4 md:grid-cols-3">
          {HOW.map((item, i) => (
            <Reveal key={item.title} delay={i * 0.06}>
              <div className="glass flex h-full flex-col rounded-2xl p-6 transition-colors hover:border-white/12">
                <div
                  className="mb-5 grid h-11 w-11 place-items-center rounded-xl"
                  style={{ background: `${item.color}1a`, color: item.color }}
                >
                  <item.icon size={21} />
                </div>
                <div className="mb-1.5 text-[12px] font-medium text-muted">{item.eyebrow}</div>
                <h3 className="mb-2 text-[19px] font-semibold">{item.title}</h3>
                <p className="flex-1 text-[14px] leading-relaxed text-ink-dim">{item.desc}</p>
                <div className="mt-5 flex items-center gap-1.5 border-t border-white/7 pt-4 text-[13px] font-medium" style={{ color: item.color }}>
                  {item.tag}
                </div>
              </div>
            </Reveal>
          ))}
        </div>

        {/* final CTA */}
        <Reveal delay={0.1}>
          <div className="glass mt-6 flex flex-col items-center gap-4 rounded-2xl p-7 sm:flex-row sm:justify-between md:p-8">
            <div>
              <p className="text-lg font-semibold">Ready to grow smarter?</p>
              <p className="mt-1 text-[13.5px] text-muted">Models run locally — no cloud account or GPU required.</p>
            </div>
            <motion.button
              onClick={onLaunch}
              className="inline-flex shrink-0 items-center gap-2 rounded-xl bg-emerald px-5 py-3 text-[14.5px] font-semibold text-[#052e16] transition-colors hover:bg-lime"
              whileTap={{ scale: 0.98 }}
            >
              Launch Platform <ArrowRight size={16} />
            </motion.button>
          </div>
        </Reveal>

        <p className="mt-8 text-center text-[12px] text-faint">
          Verdant · Vision Transformer · Sentence-Transformer · Gemma 3
        </p>
      </section>
    </motion.div>
  )
}
