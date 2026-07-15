import Link from 'next/link';

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] gap-6 text-center px-4">
      <h1 className="text-3xl font-bold tracking-tight">
        AI Music Studio
      </h1>
      <p className="text-[var(--text-muted)] max-w-md leading-relaxed">
        本地 AI 编曲工作台。从灵感到成品，四步完成一首歌的创作。
      </p>
      <div className="flex gap-4 mt-4">
        <Link
          href="/studio"
          className="px-5 py-2.5 rounded-lg bg-[var(--accent)] text-white text-sm font-medium hover:shadow-[0_0_24px_var(--accent-glow)] transition-shadow"
        >
          开始编曲
        </Link>
        <Link
          href="/library"
          className="px-5 py-2.5 rounded-lg border border-[var(--border)] text-sm text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
        >
          查看曲库
        </Link>
      </div>
    </div>
  );
}
