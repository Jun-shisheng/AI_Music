'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const links = [
  { href: '/studio', label: '编曲房' },
  { href: '/create', label: '创作' },
  { href: '/assets', label: '素材' },
  { href: '/library', label: '曲库' },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="flex items-center gap-1 px-6 py-3 border-b border-[var(--border)] bg-[var(--surface)]">
      <Link
        href="/"
        className="text-sm font-bold tracking-wider text-[var(--accent)] mr-6"
      >
        AI MUSIC
      </Link>
      {links.map((link) => {
        const active = pathname === link.href;
        return (
          <Link
            key={link.href}
            href={link.href}
            className={`px-3 py-1.5 rounded-md text-sm transition-colors ${
              active
                ? 'bg-[var(--accent)]/15 text-[var(--accent)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text)]'
            }`}
          >
            {link.label}
          </Link>
        );
      })}
    </nav>
  );
}
