'use client';

import { Suspense, useState, useEffect, useRef, useCallback } from 'react';
import { generate, getJob, audioUrl, type JobStatus } from '@/lib/api';
import { useSearchParams } from 'next/navigation';
import StructureTags from '@/components/StructureTags';
import StylePicker, { styleToPrompt, defaultStyle, parseStylePrompt } from '@/components/StylePicker';

interface FormState {
  title: string;
  lyrics: string;
  style: ReturnType<typeof defaultStyle>;
  instrumental: boolean;
}

function CreateContent() {
  const searchParams = useSearchParams();
  const lyricsRef = useRef<HTMLTextAreaElement>(null);

  const [form, setForm] = useState<FormState>({
    title: '',
    lyrics: '',
    style: defaultStyle(),
    instrumental: false,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load data from localStorage (set by studio page) or URL params
  useEffect(() => {
    const stored = localStorage.getItem('arrangeResult');
    if (stored) {
      try {
        const data = JSON.parse(stored);
        setForm((prev) => ({
          ...prev,
          title: data.title || '',
          lyrics: data.lyrics || '',
          style: data.style
            ? {
                genres: data.style.genres || [],
                moods: data.style.moods || [],
                instruments: data.style.instruments || [],
                vocals: data.style.vocals || [],
                tempo: data.style.tempo || '',
              }
            : parseStylePrompt(data.style_prompt || ''),
        }));
        localStorage.removeItem('arrangeResult');
      } catch { /* ignore */ }
    }
    // URL params override
    const lyricsParam = searchParams.get('lyrics');
    const promptParam = searchParams.get('style_prompt');
    const titleParam = searchParams.get('title');
    if (lyricsParam || promptParam || titleParam) {
      setForm((prev) => ({
        ...prev,
        title: titleParam || prev.title,
        lyrics: lyricsParam || prev.lyrics,
        style: promptParam ? parseStylePrompt(promptParam) : prev.style,
      }));
    }
  }, [searchParams]);

  // Poll job status
  useEffect(() => {
    if (!jobId) return;
    pollRef.current = setInterval(async () => {
      try {
        const j = await getJob(jobId);
        setJob(j);
        if (j.status === 'succeeded' || j.status === 'failed') {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch { /* ignore */ }
    }, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [jobId]);

  const insertTag = useCallback((tag: string) => {
    const el = lyricsRef.current;
    if (!el) return;
    const start = el.selectionStart;
    const end = el.selectionEnd;
    const before = form.lyrics.slice(0, start);
    const after = form.lyrics.slice(end);
    const insertion = before.endsWith('\n') || before === '' ? tag + '\n' : '\n' + tag + '\n';
    const newLyrics = before + insertion + after;
    setForm((f) => ({ ...f, lyrics: newLyrics }));
    // Restore cursor position after React re-render
    requestAnimationFrame(() => {
      el.focus();
      const pos = start + insertion.length;
      el.setSelectionRange(pos, pos);
    });
  }, [form.lyrics]);

  const stylePrompt = styleToPrompt(form.style);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.lyrics.trim()) return;

    setSubmitting(true);
    setError(null);
    try {
      const res = await generate({
        title: form.title.trim() || undefined,
        lyrics: form.lyrics.trim(),
        style_prompt: stylePrompt,
        instrumental: form.instrumental,
      });
      setJobId(res.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setSubmitting(false);
    }
  }

  const statusLabel: Record<string, string> = {
    queued: '排队中',
    running: '生成中',
    succeeded: '已完成',
    failed: '失败',
  };
  const statusColor: Record<string, string> = {
    queued: 'text-[var(--warning)]',
    running: 'text-sky-400 animate-pulse',
    succeeded: 'text-[var(--success)]',
    failed: 'text-[var(--error)]',
  };

  const canSubmit = form.lyrics.trim();

  return (
    <div className="max-w-5xl mx-auto p-6">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: Lyrics & Metadata */}
        <section className="lg:col-span-3 space-y-4">
          <div>
            <h2 className="text-lg font-semibold">专业创作</h2>
            <p className="text-sm text-[var(--text-muted)] mt-1">
              编辑歌词与风格，提交出歌任务。
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Title */}
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">
                歌曲标题
              </label>
              <input
                type="text"
                value={form.title}
                onChange={(e) => setForm({ ...form, title: e.target.value })}
                placeholder="可选，AI 会帮你起名"
                className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-lg p-2.5 text-sm text-[var(--text)] placeholder:text-[var(--text-muted)]/50 focus:outline-none focus:border-[var(--accent)]"
              />
            </div>

            {/* Lyrics with Structure Tags */}
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">
                歌词（含结构标签）
              </label>
              <StructureTags onInsert={insertTag} />
              <textarea
                ref={lyricsRef}
                value={form.lyrics}
                onChange={(e) => setForm({ ...form, lyrics: e.target.value })}
                className="w-full h-56 mt-1.5 bg-[var(--surface)] border border-[var(--border)] rounded-lg p-3 text-sm text-[var(--text)] font-mono resize-y focus:outline-none focus:border-[var(--accent)]"
                placeholder={`[Intro]&#10;...&#10;[Verse]&#10;...&#10;[Chorus]&#10;...&#10;[Outro]`}
              />
            </div>

            {/* Style prompt summary */}
            {stylePrompt && (
              <div>
                <label className="block text-xs text-[var(--text-muted)] mb-1">
                  风格 Prompt
                </label>
                <code className="block bg-[var(--surface)] border border-[var(--border)] rounded-lg p-2.5 text-xs text-[var(--text-muted)] leading-relaxed break-all">
                  {stylePrompt}
                </code>
              </div>
            )}

            {/* Instrumental toggle */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={form.instrumental}
                onChange={(e) => setForm({ ...form, instrumental: e.target.checked })}
                className="rounded accent-[var(--accent)]"
              />
              <span className="text-sm text-[var(--text-muted)]">纯器乐（无人声）</span>
            </label>

            {error && (
              <div className="p-3 rounded-lg bg-[var(--error)]/10 border border-[var(--error)]/30 text-sm text-[var(--error)]">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={submitting || !!jobId || !canSubmit}
              className="w-full py-2.5 rounded-lg bg-[var(--accent)] text-white text-sm font-medium disabled:opacity-40 hover:shadow-[0_0_24px_var(--accent-glow)] transition-shadow"
            >
              {submitting ? '提交中...' : jobId ? '任务已提交' : '提交出歌任务'}
            </button>
          </form>

          {/* Job Status */}
          {(jobId || job) && (
            <section className="border border-[var(--border)] rounded-lg p-4 space-y-4">
              <h3 className="text-sm font-semibold">任务状态</h3>

              {jobId && !job && (
                <p className="text-sm text-[var(--text-muted)]">提交成功，等待 Worker 处理...</p>
              )}

              {job && (
                <>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-[var(--text-muted)] font-mono">{job.id.slice(0, 8)}</span>
                    <span className={`text-sm font-medium ${statusColor[job.status]}`}>
                      {statusLabel[job.status]}
                    </span>
                    <span className="text-xs text-[var(--text-muted)]">
                      {new Date(job.updated_at).toLocaleTimeString('zh-CN')}
                    </span>
                  </div>

                  {job.error && (
                    <div className="p-3 rounded-lg bg-[var(--error)]/10 border border-[var(--error)]/30 text-sm text-[var(--error)]">
                      {job.error}
                    </div>
                  )}

                  {job.status === 'succeeded' && job.audio_path && (
                    <div className="space-y-2">
                      <p className="text-xs text-[var(--success)]">生成完成</p>
                      <audio controls className="w-full" src={audioUrl(job.audio_path)} />
                      <a
                        href={audioUrl(job.audio_path)}
                        download
                        className="inline-block text-xs text-[var(--accent)] hover:underline"
                      >
                        下载音频
                      </a>
                    </div>
                  )}

                  {job.status === 'succeeded' && (
                    <a
                      href="/library"
                      className="block text-center py-2 rounded-lg border border-[var(--border)] text-sm text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
                    >
                      前往曲库查看
                    </a>
                  )}
                </>
              )}
            </section>
          )}
        </section>

        {/* Right: Style Picker */}
        <aside className="lg:col-span-2">
          <div className="sticky top-6 border border-[var(--border)] rounded-lg p-4 bg-[var(--surface)]">
            <h3 className="text-sm font-semibold mb-4">风格配置</h3>
            <StylePicker
              value={form.style}
              onChange={(style) => setForm({ ...form, style })}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}

export default function CreatePage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center min-h-[60vh] text-sm text-[var(--text-muted)]">
        加载中...
      </div>
    }>
      <CreateContent />
    </Suspense>
  );
}
