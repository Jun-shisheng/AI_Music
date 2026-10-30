'use client';

import { useState, useMemo } from 'react';

// ── Taxonomy ──────────────────────────────────────────────────────────────

const TAXONOMY = {
  genres: [
    { value: 'pop', label: '流行' },
    { value: 'rock', label: '摇滚' },
    { value: 'folk', label: '民谣' },
    { value: 'electronic', label: '电子' },
    { value: 'hip-hop', label: '嘻哈' },
    { value: 'rnb', label: 'R&B' },
    { value: 'jazz', label: '爵士' },
    { value: 'classical', label: '古典' },
    { value: 'chinese-style', label: '国风' },
    { value: 'ancient-style', label: '古风' },
    { value: 'metal', label: '金属' },
    { value: 'punk', label: '朋克' },
    { value: 'reggae', label: '雷鬼' },
    { value: 'blues', label: '蓝调' },
    { value: 'country', label: '乡村' },
    { value: 'new-age', label: '新世纪' },
    { value: 'latin', label: '拉丁' },
    { value: 'k-pop', label: 'K-Pop' },
  ],
  moods: [
    { value: 'healing', label: '治愈' },
    { value: 'warm', label: '温暖' },
    { value: 'sad', label: '悲伤' },
    { value: 'happy', label: '快乐' },
    { value: 'romantic', label: '浪漫' },
    { value: 'passionate', label: '激情' },
    { value: 'calm', label: '平静' },
    { value: 'melancholy', label: '忧郁' },
    { value: 'epic', label: '史诗' },
    { value: 'dreamy', label: '梦幻' },
    { value: 'dark', label: '黑暗' },
    { value: 'lively', label: '活泼' },
    { value: 'nostalgic', label: '怀旧' },
    { value: 'inspirational', label: '励志' },
    { value: 'ethereal', label: '空灵' },
    { value: 'energetic', label: '活力' },
  ],
  instruments: [
    { value: 'piano', label: '钢琴' },
    { value: 'acoustic-guitar', label: '木吉他' },
    { value: 'electric-guitar', label: '电吉他' },
    { value: 'bass', label: '贝斯' },
    { value: 'drums', label: '鼓' },
    { value: 'violin', label: '小提琴' },
    { value: 'cello', label: '大提琴' },
    { value: 'flute', label: '长笛' },
    { value: 'saxophone', label: '萨克斯' },
    { value: 'guzheng', label: '古筝' },
    { value: 'erhu', label: '二胡' },
    { value: 'pipa', label: '琵琶' },
    { value: 'dizi', label: '笛子' },
    { value: 'xiao', label: '箫' },
    { value: 'synth', label: '合成器' },
    { value: 'strings', label: '弦乐组' },
    { value: 'organ', label: '管风琴' },
    { value: 'trumpet', label: '小号' },
  ],
  vocals: [
    { value: 'male', label: '男声' },
    { value: 'female', label: '女声' },
    { value: 'choir', label: '合唱' },
    { value: 'children', label: '童声' },
    { value: 'rap', label: '说唱' },
    { value: 'falsetto', label: '假声' },
    { value: 'folk-vocal', label: '民族唱法' },
    { value: 'male-female-duet', label: '男女对唱' },
  ],
};

const TEMPOS = [
  { value: '60-80', label: '慢速 60-80' },
  { value: '80-100', label: '中慢 80-100' },
  { value: '100-120', label: '中速 100-120' },
  { value: '120-140', label: '中快 120-140' },
  { value: '140-160', label: '快速 140-160' },
  { value: '160+', label: '极速 160+' },
];

const SECTION_LABELS: Record<string, string> = {
  genres: '流派',
  moods: '情绪',
  instruments: '乐器',
  vocals: '人声',
};

// ── Component ──────────────────────────────────────────────────────────────

export interface StyleState {
  genres: string[];
  moods: string[];
  instruments: string[];
  vocals: string[];
  tempo: string;
}

interface Props {
  value: StyleState;
  onChange: (v: StyleState) => void;
}

export function styleToPrompt(style: Props['value']): string {
  const parts: string[] = [];
  if (style.genres.length) parts.push(style.genres.join(', '));
  if (style.moods.length) parts.push(style.moods.join(', '));
  if (style.instruments.length) parts.push(style.instruments.join(', '));
  if (style.vocals.length) parts.push(style.vocals.join(' vocals, ') + ' vocals');
  if (style.tempo) parts.push(`${style.tempo} BPM`);
  return parts.join(', ');
}

export default function StylePicker({ value, onChange }: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    genres: true,
    moods: false,
    instruments: false,
    vocals: false,
  });

  const toggle = (section: string) =>
    setExpanded((prev) => ({ ...prev, [section]: !prev[section] }));

  const toggleTag = (section: keyof typeof TAXONOMY, tagValue: string) => {
    const current: string[] = value[section];
    const next = current.includes(tagValue)
      ? current.filter((v) => v !== tagValue)
      : [...current, tagValue];
    onChange({ ...value, [section]: next } as StyleState);
  };

  const prompt = useMemo(() => styleToPrompt(value), [value]);

  return (
    <div className="space-y-3">
      {/* Section pickers */}
      {(Object.keys(TAXONOMY) as (keyof typeof TAXONOMY)[]).map((section) => (
        <div key={section}>
          <button
            type="button"
            onClick={() => toggle(section)}
            className="flex items-center gap-1.5 w-full text-left py-1"
          >
            <span className="text-[10px] text-[var(--text-muted)] transition-transform duration-150"
              style={{ transform: expanded[section] ? 'rotate(90deg)' : '' }}
            >
              ▶
            </span>
            <span className="text-xs text-[var(--text-muted)]">
              {SECTION_LABELS[section]}
            </span>
            {value[section].length > 0 && (
              <span className="text-[10px] text-[var(--accent)]">
                ({value[section].length})
              </span>
            )}
          </button>

          {expanded[section] && (
            <div className="flex flex-wrap gap-1.5 mt-1.5 ml-4">
              {TAXONOMY[section].map((tag) => {
                const active = value[section].includes(tag.value);
                return (
                  <button
                    key={tag.value}
                    type="button"
                    onClick={() => toggleTag(section, tag.value)}
                    className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                      active
                        ? 'bg-[var(--accent)]/20 text-[var(--accent)] ring-1 ring-[var(--accent)]/40'
                        : 'bg-[var(--surface)] text-[var(--text-muted)] hover:text-[var(--text)] ring-1 ring-[var(--border)]'
                    }`}
                  >
                    {tag.label}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      ))}

      {/* Tempo */}
      <div>
        <span className="text-xs text-[var(--text-muted)]">速度</span>
        <div className="flex flex-wrap gap-1.5 mt-1.5 ml-4">
          {TEMPOS.map((t) => {
            const active = value.tempo === t.value;
            return (
              <button
                key={t.value}
                type="button"
                onClick={() =>
                  onChange({ ...value, tempo: active ? '' : t.value })
                }
                className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                  active
                    ? 'bg-sky-500/15 text-sky-400 ring-1 ring-sky-500/40'
                    : 'bg-[var(--surface)] text-[var(--text-muted)] hover:text-[var(--text)] ring-1 ring-[var(--border)]'
                }`}
              >
                {t.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Generated prompt preview */}
      {prompt && (
        <div className="pt-2 border-t border-[var(--border)]">
          <span className="text-[10px] text-[var(--text-muted)]">生成 Prompt 预览</span>
          <code className="block mt-1 bg-[var(--surface)] border border-[var(--border)] rounded p-2 text-[11px] text-[var(--text-muted)] leading-relaxed">
            {prompt}
          </code>
        </div>
      )}
    </div>
  );
}

export function defaultStyle() {
  return {
    genres: [] as string[],
    moods: [] as string[],
    instruments: [] as string[],
    vocals: [] as string[],
    tempo: '',
  };
}

/** Parse a style_prompt string back into structured style selections (best-effort). */
export function parseStylePrompt(prompt: string): ReturnType<typeof defaultStyle> {
  const result = defaultStyle();
  if (!prompt) return result;

  const lower = prompt.toLowerCase();

  for (const tag of TAXONOMY.genres) {
    if (lower.includes(tag.value)) result.genres.push(tag.value);
  }
  for (const tag of TAXONOMY.moods) {
    if (lower.includes(tag.value)) result.moods.push(tag.value);
  }
  for (const tag of TAXONOMY.instruments) {
    if (lower.includes(tag.value)) result.instruments.push(tag.value);
  }
  for (const tag of TAXONOMY.vocals) {
    if (new RegExp(`(?:^|[^a-z-])${tag.value}(?=$|[^a-z-])`).test(lower)) {
      result.vocals.push(tag.value);
    }
  }

  const tempoMatch = prompt.match(/(\d{2,3}\s*[-~]\s*\d{2,3}|\d{2,3}\s*\+)\s*BPM/i);
  if (tempoMatch) {
    const tempo = tempoMatch[1].replace(/\s/g, '').replace('~', '-');
    result.tempo = TEMPOS.find((t) => t.value === tempo)?.value || '';
  }

  return result;
}
