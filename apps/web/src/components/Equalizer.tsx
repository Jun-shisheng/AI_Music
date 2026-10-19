interface Props {
  bars?: number;
  className?: string;
  /** Pause the animation (e.g. a finished job). */
  idle?: boolean;
}

/** Deterministic pseudo-random heights so server and client markup agree. */
function heightFor(index: number) {
  const n = Math.sin(index * 12.9898) * 43758.5453;
  return 24 + Math.round((n - Math.floor(n)) * 66);
}

export default function Equalizer({ bars = 36, className = '', idle = false }: Props) {
  return (
    <div
      className={`flex items-center gap-[3px] ${className}`}
      role="img"
      aria-label={idle ? '静态波形' : '播放中的波形'}
    >
      {Array.from({ length: bars }, (_, i) => {
        const height = heightFor(i);
        return (
          <span
            key={i}
            className="eq-bar w-[3px] rounded-full bg-gradient-to-t from-[var(--accent)] to-[var(--accent-2)]"
            style={{
              height: `${height}%`,
              animationDelay: `${(i % 12) * 0.09}s`,
              animationDuration: `${1 + (i % 5) * 0.18}s`,
              animationPlayState: idle ? 'paused' : 'running',
            }}
          />
        );
      })}
    </div>
  );
}
