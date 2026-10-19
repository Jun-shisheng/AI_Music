import Link from 'next/link';
import Equalizer from '@/components/Equalizer';
import {
  GamepadIcon,
  LayersIcon,
  LibraryIcon,
  SlidersIcon,
  SparklesIcon,
} from '@/components/Icons';

const steps = [
  {
    href: '/studio',
    step: '01',
    title: '编曲房',
    description: '描述想法，LLM 生成结构化歌词、风格标签与生成 Prompt。',
    icon: SparklesIcon,
  },
  {
    href: '/create',
    step: '02',
    title: '专业创作',
    description: '手动精修歌词、结构标签与配器，提交本地模型出歌。',
    icon: SlidersIcon,
  },
  {
    href: '/assets',
    step: '03',
    title: '游戏素材',
    description: 'BGM 循环与短音效，一次生成多个候选版本。',
    icon: GamepadIcon,
  },
  {
    href: '/library',
    step: '04',
    title: '曲库',
    description: '试听、筛选、下载所有已生成和正在生成的歌曲。',
    icon: LibraryIcon,
  },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4 pb-20 pt-10 sm:px-6 sm:pt-16">
      {/* Hero */}
      <section className="grid items-center gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:gap-14">
        <div className="animate-fade-up">
          <p className="eyebrow flex items-center gap-2">
            <span className="h-px w-8 bg-gradient-to-r from-[var(--accent)] to-transparent" />
            Local-first · GPU audio
          </p>
          <h1 className="display-title mt-5 text-4xl font-bold leading-[1.12] tracking-tight sm:text-5xl lg:text-[3.4rem]">
            把灵感，
            <br />
            变成一首完整的歌。
          </h1>
          <p className="mt-5 max-w-lg text-[15px] leading-relaxed text-[var(--text-muted)]">
            本地 AI 编曲工作台。LLM 写词、本地模型渲染、指标测量后人工定稿——
            四步走完一首歌，音频不离开这台机器。
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link href="/studio" className="btn-primary">
              <SparklesIcon className="h-4 w-4" />
              开始编曲
            </Link>
            <Link href="/library" className="btn-ghost">
              <LibraryIcon className="h-4 w-4" />
              打开曲库
            </Link>
          </div>

          <dl className="mt-10 flex flex-wrap gap-x-8 gap-y-3">
            {[
              ['4', '步工作流'],
              ['3', '种生成类型'],
              ['0', '云端上传'],
            ].map(([value, label]) => (
              <div key={label}>
                <dt className="text-xl font-semibold tabular-nums">{value}</dt>
                <dd className="mt-0.5 text-[11px] tracking-wide text-[var(--text-muted)]">
                  {label}
                </dd>
              </div>
            ))}
          </dl>
        </div>

        {/* Visual */}
        <div className="animate-fade-up [animation-delay:120ms]">
          <div className="panel relative overflow-hidden p-5 sm:p-6">
            <div className="pointer-events-none absolute -right-16 -top-20 h-56 w-56 rounded-full bg-[var(--accent)]/20 blur-3xl" />
            <div className="pointer-events-none absolute -bottom-24 -left-10 h-48 w-48 rounded-full bg-[var(--accent-2)]/10 blur-3xl" />

            <div className="relative flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="live-dot h-1.5 w-1.5 rounded-full bg-[var(--success)]" />
                <span className="text-[10px] font-medium uppercase tracking-[0.2em] text-[var(--text-muted)]">
                  Now rendering
                </span>
              </div>
              <span className="rounded-md border border-[var(--border)] px-2 py-0.5 text-[10px] text-[var(--text-muted)]">
                48 kHz · WAV
              </span>
            </div>

            <div className="relative mt-6 flex h-36 items-center justify-center rounded-xl border border-[var(--border)] bg-[rgba(6,6,12,0.55)] px-4">
              <Equalizer bars={42} className="h-24 w-full justify-center" />
            </div>

            <div className="relative mt-5 flex items-start justify-between gap-4">
              <div>
                <p className="text-base font-semibold">未命名作品</p>
                <p className="mt-1 text-xs text-[var(--text-muted)]">
                  chinese-style · healing · female vocals
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs tabular-nums text-[var(--text)]">02:47</p>
                <p className="mt-1 text-[10px] text-[var(--text-muted)]">rendering</p>
              </div>
            </div>

            <div className="relative mt-4 h-1 overflow-hidden rounded-full bg-white/[0.06]">
              <div className="h-full w-2/3 rounded-full bg-gradient-to-r from-[var(--accent)] to-[var(--accent-2)]" />
            </div>
          </div>
        </div>
      </section>

      {/* Steps */}
      <section className="mt-20 sm:mt-24">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Workflow</p>
            <h2 className="mt-2 text-xl font-semibold tracking-tight sm:text-2xl">
              从灵感到交付，四步完成
            </h2>
          </div>
          <LayersIcon className="hidden h-6 w-6 text-[var(--text-muted)] sm:block" />
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {steps.map(({ href, step, title, description, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className="panel panel-hover group flex flex-col p-5"
            >
              <div className="flex items-center justify-between">
                <span className="grid h-10 w-10 place-items-center rounded-xl border border-[var(--border)] bg-[rgba(10,10,18,0.5)] text-[var(--accent-soft)] transition-colors duration-200 group-hover:border-[var(--accent)]/40 group-hover:text-[var(--accent-2)]">
                  <Icon className="h-5 w-5" />
                </span>
                <span className="text-[10px] font-medium tabular-nums tracking-[0.2em] text-[var(--text-muted)]">
                  {step}
                </span>
              </div>
              <h3 className="mt-5 text-sm font-semibold">{title}</h3>
              <p className="mt-2 text-xs leading-relaxed text-[var(--text-muted)]">
                {description}
              </p>
              <span className="mt-4 text-[11px] text-[var(--accent-soft)] opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                进入 →
              </span>
            </Link>
          ))}
        </div>
      </section>

      {/* Footer note */}
      <section className="mt-16 flex flex-col items-center gap-3 border-t border-[var(--border)] pt-8 text-center">
        <Equalizer bars={20} idle className="h-6 opacity-40" />
        <p className="text-[11px] tracking-wide text-[var(--text-muted)]">
          音频、模型权重与歌曲数据都保存在本机 · 只有可选的写词步骤会调用外部 LLM
        </p>
      </section>
    </div>
  );
}
