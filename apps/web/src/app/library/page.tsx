'use client';

import { useState, useEffect, useRef } from 'react';
import { getLibrary, audioUrl, type LibraryItem } from '@/lib/api';
import { PauseIcon, PlayIcon, RefreshIcon } from '@/components/Icons';

const statusLabel: Record<string, string> = {
  queued: '排队中',
  running: '生成中',
  succeeded: '已完成',
  failed: '失败',
};
const statusColor: Record<string, string> = {
  queued: 'text-[var(--warning)]',
  running: 'text-sky-400',
  succeeded: 'text-[var(--success)]',
  failed: 'text-[var(--error)]',
};

export default function LibraryPage() {
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const data = await getLibrary(filter || undefined);
      setItems(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Load failed');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, [filter]);

  // Changing the src of a mounted <audio> does not re-trigger autoplay, so
  // switching from one track to another needs an explicit play().
  useEffect(() => {
    if (playing && audioRef.current) {
      audioRef.current.play().catch(() => {
        /* the browser may reject autoplay; the controls still work */
      });
    }
  }, [playing]);

  // Poll when there are active jobs
  useEffect(() => {
    const hasActive = items.some(
      (i) => i.status === 'queued' || i.status === 'running'
    );
    if (!hasActive) return;

    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, [items]);

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="eyebrow">Library</p>
          <h2 className="mt-2 text-xl font-semibold tracking-tight">曲库</h2>
          <p className="mt-1.5 text-sm text-[var(--text-muted)]">
            所有已生成和正在生成的歌曲。
          </p>
        </div>
        <button onClick={load} className="btn-ghost px-3 py-1.5 text-xs">
          <RefreshIcon className="h-3.5 w-3.5" />
          刷新
        </button>
      </div>

      {/* Filter */}
      <div className="flex gap-2">
        {['', 'succeeded', 'running', 'queued', 'failed'].map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`rounded-lg px-3 py-1 text-xs transition-colors ${
              filter === s
                ? 'bg-[var(--accent)]/15 text-[var(--accent-soft)] ring-1 ring-[var(--accent)]/40'
                : 'text-[var(--text-muted)] ring-1 ring-transparent hover:text-[var(--text)]'
            }`}
          >
            {s ? statusLabel[s] : '全部'}
          </button>
        ))}
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-[var(--error)]/10 border border-[var(--error)]/30 text-sm text-[var(--error)]">
          {error}
        </div>
      )}

      {loading && items.length === 0 && (
        <div className="flex items-center justify-center h-48 text-sm text-[var(--text-muted)]">
          加载中...
        </div>
      )}

      {!loading && items.length === 0 && (
        <div className="flex h-48 flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-[var(--border)] bg-white/[0.015] text-sm text-[var(--text-muted)]">
          <p>曲库为空</p>
          <a href="/studio" className="text-[var(--accent)] hover:underline">
            去编曲房创作第一首歌
          </a>
        </div>
      )}

      {/* Track List */}
      {items.length > 0 && (
        <div className="space-y-2">
          {items.map((item) => (
            <div
              key={item.id}
              className="panel panel-hover flex items-center gap-4 p-4"
            >
              {/* Status dot */}
              <span
                className={`w-2 h-2 rounded-full ${
                  item.status === 'succeeded'
                    ? 'bg-[var(--success)]'
                    : item.status === 'failed'
                    ? 'bg-[var(--error)]'
                    : item.status === 'running'
                    ? 'bg-sky-400 animate-pulse'
                    : 'bg-[var(--warning)]'
                }`}
              />

              {/* Info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">
                  {item.title || '未命名'}
                </p>
                <div className="flex items-center gap-2 text-xs">
                  <span className={statusColor[item.status]}>
                    {statusLabel[item.status]}
                  </span>
                  <span className="text-[var(--text-muted)]">
                    {new Date(item.created_at).toLocaleString('zh-CN')}
                  </span>
                </div>
              </div>

              {/* Actions */}
              {item.status === 'succeeded' && item.audio_path && (
                <button
                  onClick={() => {
                    if (playing === item.id) {
                      audioRef.current?.pause();
                      setPlaying(null);
                    } else {
                      setPlaying(item.id);
                    }
                  }}
                  className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-[var(--accent)]/50 px-3 py-1.5 text-xs font-medium text-[var(--accent-soft)] transition-colors hover:border-[var(--accent)] hover:bg-[var(--accent)]/10"
                >
                  {playing === item.id ? (
                    <>
                      <PauseIcon className="h-3 w-3" /> 暂停
                    </>
                  ) : (
                    <>
                      <PlayIcon className="h-3 w-3" /> 试听
                    </>
                  )}
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Hidden audio element */}
      {playing && (
        <audio
          ref={audioRef}
          autoPlay
          onEnded={() => setPlaying(null)}
          src={audioUrl(
            items.find((i) => i.id === playing)?.audio_url ||
              items.find((i) => i.id === playing)?.audio_path ||
              ''
          )}
          className="hidden"
        />
      )}
    </div>
  );
}
