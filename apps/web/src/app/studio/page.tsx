'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { arrange, type ArrangeResponse } from '@/lib/api';

const STYLE_COLORS: Record<string, string> = {
  genres: 'bg-[var(--accent)]/15 text-[var(--accent)]',
  moods: 'bg-emerald-500/10 text-emerald-400',
  instruments: 'bg-amber-500/10 text-amber-400',
  vocals: 'bg-rose-500/10 text-rose-400',
  tempo: 'bg-sky-500/10 text-sky-400',
};

const SECTION_LABELS: Record<string, string> = {
  genres: '流派',
  moods: '情绪',
  instruments: '乐器',
  vocals: '人声',
  tempo: '速度',
};

export default function StudioPage() {
  const router = useRouter();
  const [prompt, setPrompt] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ArrangeResponse | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await arrange({
        prompt: prompt.trim(),
        lyrics: lyrics.trim() || undefined,
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  function handleGoCreate() {
    if (!result) return;
    localStorage.setItem('arrangeResult', JSON.stringify(result));
    router.push('/create');
  }

  return (
    <div className="max-w-5xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input Panel */}
      <section className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold">编曲房</h2>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            描述你想创作的歌曲，AI 会生成结构化歌词与风格标签。
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">
              创作要求
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="例：一首关于夏天的国风流行歌，钢琴伴奏，女声，治愈温暖"
              className="w-full h-28 bg-[var(--surface)] border border-[var(--border)] rounded-lg p-3 text-sm text-[var(--text)] placeholder:text-[var(--text-muted)]/50 resize-none focus:outline-none focus:border-[var(--accent)]"
            />
          </div>

          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">
              已有歌词（可选）
            </label>
            <textarea
              value={lyrics}
              onChange={(e) => setLyrics(e.target.value)}
              placeholder="粘贴已有歌词，AI 会自动添加结构标签并优化..."
              className="w-full h-32 bg-[var(--surface)] border border-[var(--border)] rounded-lg p-3 text-sm text-[var(--text)] placeholder:text-[var(--text-muted)]/50 resize-none focus:outline-none focus:border-[var(--accent)]"
            />
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-[var(--error)]/10 border border-[var(--error)]/30 text-sm text-[var(--error)]">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !prompt.trim()}
            className="w-full py-2.5 rounded-lg bg-[var(--accent)] text-white text-sm font-medium disabled:opacity-40 hover:shadow-[0_0_24px_var(--accent-glow)] transition-shadow"
          >
            {loading ? (
              <span className="inline-flex items-center gap-2">
                <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                正在调用 DeepSeek...
              </span>
            ) : (
              '开始编排'
            )}
          </button>

          <p className="text-[10px] text-[var(--text-muted)] text-center">
            Ctrl + Enter 快速提交
          </p>
        </form>
      </section>

      {/* Result Panel */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold">编排结果</h2>

        {!result && !loading && (
          <div className="flex items-center justify-center h-64 text-sm text-[var(--text-muted)] border border-dashed border-[var(--border)] rounded-lg">
            等待输入
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center h-64 gap-3 text-sm text-[var(--text-muted)] border border-dashed border-[var(--border)] rounded-lg">
            <span className="w-6 h-6 border-2 border-[var(--accent)]/30 border-t-[var(--accent)] rounded-full animate-spin" />
            DeepSeek 编排中...
          </div>
        )}

        {result && (
          <div className="space-y-4">
            {/* Title */}
            {result.title && (
              <div>
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Title</p>
                <h3 className="text-xl font-bold mt-0.5">{result.title}</h3>
              </div>
            )}

            {/* Lyrics */}
            <div>
              <p className="text-xs text-[var(--text-muted)] mb-1.5">歌词</p>
              <pre className="bg-[var(--surface)] border border-[var(--border)] rounded-lg p-3 text-sm whitespace-pre-wrap leading-relaxed max-h-72 overflow-y-auto font-mono text-[var(--text)]">
                {result.lyrics}
              </pre>
            </div>

            {/* Style Tags */}
            <div>
              <p className="text-xs text-[var(--text-muted)] mb-2">风格标签</p>
              {(['genres', 'moods', 'instruments', 'vocals'] as const).map((cat) => {
                const values = result.style[cat];
                if (!values || values.length === 0) return null;
                return (
                  <div key={cat} className="mb-1.5">
                    <span className="text-[10px] text-[var(--text-muted)] mr-2">
                      {SECTION_LABELS[cat]}
                    </span>
                    {values.map((v: string) => (
                      <span
                        key={v}
                        className={`inline-block px-1.5 py-0.5 rounded text-[10px] mr-1 ${STYLE_COLORS[cat]}`}
                      >
                        {v}
                      </span>
                    ))}
                  </div>
                );
              })}
              {result.style.tempo && (
                <div>
                  <span className="text-[10px] text-[var(--text-muted)] mr-2">
                    {SECTION_LABELS.tempo}
                  </span>
                  <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] ${STYLE_COLORS.tempo}`}>
                    {result.style.tempo}
                  </span>
                </div>
              )}
            </div>

            {/* Style Prompt */}
            <div>
              <p className="text-xs text-[var(--text-muted)] mb-1">生成 Prompt</p>
              <code className="block bg-[var(--surface)] border border-[var(--border)] rounded-lg p-2 text-xs text-[var(--text-muted)] break-all">
                {result.style_prompt}
              </code>
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-2">
              <button
                onClick={handleGoCreate}
                className="flex-1 py-2 rounded-lg bg-[var(--accent)] text-white text-sm font-medium hover:shadow-[0_0_24px_var(--accent-glow)] transition-shadow"
              >
                前往创作页面微调并生成
              </button>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(JSON.stringify(result, null, 2));
                }}
                className="px-3 py-2 rounded-lg border border-[var(--border)] text-xs text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
                title="复制 JSON"
              >
                复制
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
