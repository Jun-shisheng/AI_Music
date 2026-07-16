'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { audioUrl, generate, getJob, type JobStatus } from '@/lib/api';

type AssetMode = 'bgm_loop' | 'sfx';

const MODES: Record<AssetMode, {
  label: string;
  description: string;
  defaultDuration: number;
  loopable: boolean;
  prompt: string;
}> = {
  bgm_loop: {
    label: 'BGM 循环',
    description: '适合地图、关卡、战斗前氛围和菜单背景。',
    defaultDuration: 30,
    loopable: true,
    prompt: 'instrumental game background music, seamless loop, clean loop point, no vocals',
  },
  sfx: {
    label: '短音效',
    description: '适合打击、拾取、UI、技能释放等素材草案。',
    defaultDuration: 10,
    loopable: false,
    prompt: 'short game sound effect, clean transient, no melody, no vocals',
  },
};

const WORLDS = [
  {
    id: 'elf',
    label: '精灵世界',
    prompt: 'fantasy elf forest, airy, magical, harp, flute, soft bells, organic ambience',
  },
  {
    id: 'apocalypse',
    label: '末日废土',
    prompt: 'post-apocalyptic wasteland, tense, industrial drones, metal impacts, dusty atmosphere',
  },
  {
    id: 'cyber',
    label: '赛博都市',
    prompt: 'cyberpunk city, neon, dark synth, pulsing bass, glitch textures, futuristic',
  },
  {
    id: 'ancient',
    label: '古风秘境',
    prompt: 'ancient eastern fantasy, guzheng, dizi, erhu, misty mountains, restrained percussion',
  },
  {
    id: 'horror',
    label: '恐怖悬疑',
    prompt: 'horror suspense, low drones, distant whispers, dissonant strings, uneasy silence',
  },
];

const SFX_HINTS = [
  'heavy sword impact',
  'magic pickup sparkle',
  'UI confirm click',
  'fire spell cast',
  'stone door open',
  'enemy hit feedback',
];

function statusText(status: string) {
  const labels: Record<string, string> = {
    queued: '排队中',
    running: '生成中',
    succeeded: '已完成',
    failed: '失败',
  };
  return labels[status] || status;
}

function statusClass(status: string) {
  if (status === 'succeeded') return 'text-[var(--success)]';
  if (status === 'failed') return 'text-[var(--error)]';
  if (status === 'running') return 'text-sky-400';
  return 'text-[var(--warning)]';
}

export default function AssetsPage() {
  const [mode, setMode] = useState<AssetMode>('bgm_loop');
  const [worldId, setWorldId] = useState('elf');
  const [brief, setBrief] = useState('calm forest village exploration loop');
  const [duration, setDuration] = useState(MODES.bgm_loop.defaultDuration);
  const [variations, setVariations] = useState(4);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobIds, setJobIds] = useState<string[]>([]);
  const [jobs, setJobs] = useState<Record<string, JobStatus>>({});
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const world = WORLDS.find((item) => item.id === worldId) || WORLDS[0];
  const modeSpec = MODES[mode];

  const promptPreview = useMemo(() => {
    return [
      world.prompt,
      modeSpec.prompt,
      brief.trim(),
      mode === 'bgm_loop' ? `${duration} seconds, loopable` : `${duration} seconds, isolated usable sound`,
    ].filter(Boolean).join(', ');
  }, [brief, duration, mode, modeSpec.prompt, world.prompt]);

  useEffect(() => {
    setDuration(MODES[mode].defaultDuration);
    if (mode === 'sfx' && brief === 'calm forest village exploration loop') {
      setBrief('heavy sword impact with sharp attack and short tail');
    }
    if (mode === 'bgm_loop' && brief === 'heavy sword impact with sharp attack and short tail') {
      setBrief('calm forest village exploration loop');
    }
  }, [mode, brief]);

  useEffect(() => {
    if (!jobIds.length) return;

    async function poll() {
      const next: Record<string, JobStatus> = {};
      await Promise.all(jobIds.map(async (id) => {
        try {
          next[id] = await getJob(id);
        } catch {
          if (jobs[id]) next[id] = jobs[id];
        }
      }));
      setJobs((prev) => ({ ...prev, ...next }));
    }

    poll();
    timerRef.current = setInterval(poll, 3000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [jobIds]);

  const orderedJobs = jobIds.map((id) => jobs[id]).filter(Boolean);
  const hasActive = orderedJobs.some((job) => job.status === 'queued' || job.status === 'running');

  useEffect(() => {
    if (!hasActive && timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, [hasActive]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!brief.trim()) return;

    setSubmitting(true);
    setError(null);
    setJobs({});
    setJobIds([]);

    try {
      const ids: string[] = [];
      for (let i = 0; i < variations; i += 1) {
        const label = String.fromCharCode(65 + i);
        const response = await generate({
          title: `${world.label} ${modeSpec.label} ${label}`,
          lyrics: '[Instrumental]',
          style_prompt: `${promptPreview}, variation ${label}, distinct interpretation`,
          instrumental: true,
          generation_type: mode,
          duration_seconds: duration,
          loopable: modeSpec.loopable,
          variation_label: label,
        });
        ids.push(response.id);
      }
      setJobIds(ids);
    } catch (err) {
      setError(err instanceof Error ? err.message : '提交失败');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-5 gap-6">
      <section className="lg:col-span-2 space-y-5">
        <div>
          <h2 className="text-lg font-semibold">游戏素材工坊</h2>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            批量生成独游用 BGM 循环和短音效草案，再从曲库中挑选可用版本。
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="grid grid-cols-2 gap-2">
            {(Object.keys(MODES) as AssetMode[]).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setMode(key)}
                className={`text-left p-3 rounded-lg border transition-colors ${
                  mode === key
                    ? 'border-[var(--accent)] bg-[var(--accent)]/10'
                    : 'border-[var(--border)] bg-[var(--surface)] hover:border-[var(--accent)]/50'
                }`}
              >
                <span className="block text-sm font-medium">{MODES[key].label}</span>
                <span className="block text-xs text-[var(--text-muted)] mt-1">
                  {MODES[key].description}
                </span>
              </button>
            ))}
          </div>

          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-2">世界观 Agent</label>
            <div className="flex flex-wrap gap-2">
              {WORLDS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setWorldId(item.id)}
                  className={`px-3 py-1 rounded-md text-xs transition-colors ${
                    worldId === item.id
                      ? 'bg-[var(--accent)]/15 text-[var(--accent)]'
                      : 'bg-[var(--surface)] text-[var(--text-muted)] ring-1 ring-[var(--border)] hover:text-[var(--text)]'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">素材用途</label>
            <textarea
              value={brief}
              onChange={(e) => setBrief(e.target.value)}
              className="w-full h-24 bg-[var(--surface)] border border-[var(--border)] rounded-lg p-3 text-sm text-[var(--text)] resize-none focus:outline-none focus:border-[var(--accent)]"
              placeholder="例如：heavy sword impact / forest village exploration loop"
            />
            {mode === 'sfx' && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {SFX_HINTS.map((hint) => (
                  <button
                    key={hint}
                    type="button"
                    onClick={() => setBrief(hint)}
                    className="px-2 py-0.5 rounded text-[11px] bg-[var(--surface)] text-[var(--text-muted)] ring-1 ring-[var(--border)] hover:text-[var(--accent)]"
                  >
                    {hint}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="block text-xs text-[var(--text-muted)] mb-1">时长</span>
              <select
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-lg p-2 text-sm"
              >
                {(mode === 'sfx' ? [10, 15, 20] : [30, 45, 60, 90, 120]).map((value) => (
                  <option key={value} value={value}>{value}s</option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="block text-xs text-[var(--text-muted)] mb-1">变体数量</span>
              <select
                value={variations}
                onChange={(e) => setVariations(Number(e.target.value))}
                className="w-full bg-[var(--surface)] border border-[var(--border)] rounded-lg p-2 text-sm"
              >
                {[2, 3, 4, 5, 6].map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>

          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">Prompt 预览</label>
            <code className="block bg-[var(--surface)] border border-[var(--border)] rounded-lg p-3 text-xs text-[var(--text-muted)] leading-relaxed">
              {promptPreview}
            </code>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-[var(--error)]/10 border border-[var(--error)]/30 text-sm text-[var(--error)]">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting || !brief.trim()}
            className="w-full py-2.5 rounded-lg bg-[var(--accent)] text-white text-sm font-medium disabled:opacity-40 hover:shadow-[0_0_24px_var(--accent-glow)] transition-shadow"
          >
            {submitting ? '提交中...' : `生成 ${variations} 个变体`}
          </button>
        </form>
      </section>

      <section className="lg:col-span-3 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">候选素材</h2>
          <a href="/library" className="text-xs text-[var(--accent)] hover:underline">
            打开曲库
          </a>
        </div>

        {!jobIds.length && (
          <div className="h-72 flex items-center justify-center border border-dashed border-[var(--border)] rounded-lg text-sm text-[var(--text-muted)]">
            提交后会在这里显示所有变体的生成状态和试听入口。
          </div>
        )}

        {jobIds.length > 0 && (
          <div className="space-y-2">
            {jobIds.map((id, index) => {
              const job = jobs[id];
              const label = String.fromCharCode(65 + index);
              return (
                <div key={id} className="p-4 rounded-lg bg-[var(--surface)] border border-[var(--border)] space-y-3">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-sm font-medium">
                        变体 {label}
                        {job?.duration_seconds ? ` · ${job.duration_seconds}s` : ''}
                        {job?.loopable ? ' · Loop' : ''}
                      </p>
                      <p className="text-xs text-[var(--text-muted)] mt-1 font-mono">{id}</p>
                    </div>
                    <span className={`text-xs ${statusClass(job?.status || 'queued')}`}>
                      {statusText(job?.status || 'queued')}
                    </span>
                  </div>

                  {job?.error && (
                    <div className="p-2 rounded bg-[var(--error)]/10 text-xs text-[var(--error)]">
                      {job.error}
                    </div>
                  )}

                  {job?.status === 'succeeded' && (job.audio_url || job.audio_path) && (
                    <div className="space-y-2">
                      <audio controls className="w-full" src={audioUrl(job.audio_url || job.audio_path || '')} />
                      <a
                        href={audioUrl(job.audio_url || job.audio_path || '')}
                        download
                        className="inline-block text-xs text-[var(--accent)] hover:underline"
                      >
                        下载 WAV
                      </a>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
