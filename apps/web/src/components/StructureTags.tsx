'use client';

const STRUCTURE_TAGS = [
  { tag: '[Intro]', label: '前奏' },
  { tag: '[Verse]', label: '主歌' },
  { tag: '[Pre-Chorus]', label: '预副歌' },
  { tag: '[Chorus]', label: '副歌' },
  { tag: '[Bridge]', label: '桥段' },
  { tag: '[Solo]', label: '间奏' },
  { tag: '[Outro]', label: '尾奏' },
  { tag: '[Hook]', label: 'Hook' },
  { tag: '[Rap]', label: '说唱' },
  { tag: '[Drop]', label: 'Drop' },
];

interface Props {
  onInsert: (tag: string) => void;
}

export default function StructureTags({ onInsert }: Props) {
  return (
    <div className="flex items-center gap-1 flex-wrap">
      <span className="text-[10px] text-[var(--text-muted)] mr-1">结构:</span>
      {STRUCTURE_TAGS.map((st) => (
        <button
          key={st.tag}
          type="button"
          onClick={() => onInsert(st.tag)}
          className="px-1.5 py-0.5 rounded text-[10px] bg-[var(--surface)] text-[var(--text-muted)] ring-1 ring-[var(--border)] hover:text-[var(--accent)] hover:ring-[var(--accent)]/40 transition-colors"
          title={`插入 ${st.tag}`}
        >
          {st.label}
        </button>
      ))}
    </div>
  );
}
