'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { WaveformIcon } from '@/components/Icons';

const links = [
  { href: '/studio', label: '编曲房' },
  { href: '/create', label: '创作' },
  { href: '/assets', label: '素材' },
  { href: '/library', label: '曲库' },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[rgba(6,6,11,0.72)] backdrop-blur-xl">
      <nav className="mx-auto flex h-16 max-w-6xl items-center gap-2 px-4 sm:px-6">
        <Link href="/" className="group mr-3 flex shrink-0 items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-[var(--accent)] to-[#4c1d95] shadow-lg shadow-[var(--accent-glow)] transition-transform duration-200 group-hover:-rotate-6">
            <WaveformIcon className="h-5 w-5 text-white" />
          </span>
          <span className="flex flex-col leading-none">
            <span className="text-sm font-bold tracking-[0.16em]">AI MUSIC</span>
            <span className="mt-1 text-[9px] tracking-[0.34em] text-[var(--text-muted)]">
              STUDIO
            </span>
          </span>
        </Link>

        <div className="flex flex-1 items-center gap-1 overflow-x-auto">
          {links.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`relative shrink-0 rounded-lg px-3 py-2 text-sm transition-colors duration-200 ${
                  active
                    ? 'bg-white/[0.06] text-[var(--text)]'
                    : 'text-[var(--text-muted)] hover:bg-white/[0.03] hover:text-[var(--text)]'
                }`}
              >
                {link.label}
                {active && (
                  <span className="absolute inset-x-3 -bottom-[13px] h-px bg-gradient-to-r from-transparent via-[var(--accent-soft)] to-transparent" />
                )}
              </Link>
            );
          })}
        </div>

        <div className="hidden shrink-0 items-center gap-2 rounded-full border border-[var(--border)] bg-[rgba(10,10,18,0.5)] px-3 py-1.5 sm:flex">
          <span className="live-dot h-1.5 w-1.5 rounded-full bg-[var(--success)]" />
          <span className="text-[10px] tracking-wide text-[var(--text-muted)]">本地运行</span>
        </div>
      </nav>
    </header>
  );
}
