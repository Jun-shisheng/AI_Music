"""Curate a Chinese batch into a clean, dated delivery folder.

The batch's `selected/` is "the best take the ranker found", not "good enough to
deliver".  This tool applies the human-facing gates and copies only the takes
that clear them:

  * lyric gate passes (score >= 80, no critical issue),
  * symbolic melody gate passes (density / register / hook / range / climax),
  * audio arc is `build-up`, lift >= --min-lift, coverage >= --min-coverage,
  * duration inside 2:00-3:30.

Everything else is listed in `excluded.md` with the reason, so a weak take is
not silently shipped and not silently deleted.

Output layout (the project's settled convention):
  <delivery>/<slug>_YuE2.wav
  <delivery>/<slug>_YuE2.txt   (header names engine + that take's numbers)
  <delivery>/manifest.json
  <delivery>/README.md
  <delivery>/excluded.md

Usage:
  P=vendor/ACE-Step-1.5/.venv/Scripts/python.exe
  $P -I tools/generation/zh_pop_delivery.py \
      --out-dir outputs/2026-10-26_zh_pop \
      --inputs inputs/2026-10-26_zh_pop \
      --delivery outputs/2026-10-26_zh_pop/delivery \
      --include auto
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "generation"))
from melody_qa import audit as melody_audit  # noqa: E402
from zh_lyric_qa import audit as lyric_audit  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def melody_for(out_dir: Path, take_stem: str) -> dict | None:
    matches = list((out_dir / "candidates").glob(
        f"*/{take_stem}_artifacts/score.abc"))
    if not matches:
        return None
    try:
        return asdict(melody_audit(matches[0].read_text(encoding="utf-8-sig"),
                                   target="zh-pop"))
    except Exception:
        return None


def fmt(value, digits=1, suffix=""):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs" / "2026-10-26_zh_pop")
    ap.add_argument("--inputs", type=Path,
                    default=ROOT / "inputs" / "2026-10-26_zh_pop")
    ap.add_argument("--delivery", type=Path,
                    default=ROOT / "outputs" / "2026-10-26_zh_pop" / "delivery")
    ap.add_argument("--include", default="auto",
                    help="'auto' applies the gates; or a comma-separated slug list")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--min-lift", type=float, default=8.0)
    ap.add_argument("--min-coverage", type=float, default=0.85)
    ap.add_argument("--min-duration", type=float, default=120.0)
    ap.add_argument("--max-duration", type=float, default=210.0)
    args = ap.parse_args()

    titles = load_json(args.inputs / "_titles.json", {})
    modes = load_json(args.inputs / "_modes.json", {})
    index = {row.get("title"): row for row in
             load_json(args.out_dir / "index.json", [])}
    include = None if args.include == "auto" else {
        s.strip() for s in args.include.split(",") if s.strip()}
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    rows = []
    for slug, title in titles.items():
        row = index.get(slug)
        if not row:
            rows.append({"slug": slug, "title": title, "reasons": ["未出片"]})
            continue
        if include is not None and slug not in include:
            continue
        if slug in exclude:
            rows.append({"slug": slug, "title": title, "reasons": ["显式排除"]})
            continue
        lyrics_path = args.inputs / f"{slug}_lyrics.txt"
        if not lyrics_path.exists():
            rows.append({"slug": slug, "title": title, "reasons": ["缺少歌词"]})
            continue
        text = lyrics_path.read_text(encoding="utf-8")
        qa = lyric_audit(text, title=title, mode=modes.get(slug, "narrative"))
        melody = melody_for(args.out_dir, Path(row.get("selected", "")).stem)
        arc = row.get("arc") or {}
        coverage = row.get("diction")
        duration = float(arc.get("duration_s") or 0.0)

        reasons: list[str] = []
        if not qa.passes:
            reasons.append(f"词检 {qa.score} 未过")
        if melody and not melody.get("passes"):
            reasons.append(f"旋律 {melody.get('score')} 未过")
        elif melody is None:
            reasons.append("缺少旋律计划")
        if arc.get("verdict") != "build-up":
            reasons.append(f"动态 {arc.get('verdict')}")
        if float(arc.get("lift_db") or 0.0) < args.min_lift:
            reasons.append(f"起伏 {fmt(arc.get('lift_db'), 1, ' dB')} < "
                           f"{args.min_lift:g} dB")
        if coverage is not None and float(coverage) < args.min_coverage:
            reasons.append(f"歌词覆盖 {coverage:.2f} < {args.min_coverage:g}")
        if not args.min_duration <= duration <= args.max_duration:
            reasons.append(f"时长 {duration:.0f}s 不在 "
                           f"{args.min_duration:.0f}-{args.max_duration:.0f}s")

        rows.append({
            "slug": slug, "title": title, "row": row, "qa": qa,
            "melody": melody, "arc": arc, "coverage": coverage,
            "duration": duration, "lyrics": text, "reasons": reasons,
        })

    good = [r for r in rows if "row" in r and not r["reasons"]]
    bad = [r for r in rows if r.get("reasons")]
    if not good:
        print("no take clears the delivery gates", file=sys.stderr)
        for r in bad:
            print(f"  {r['slug']}: {'; '.join(r['reasons'])}", file=sys.stderr)
        return 1

    if args.delivery.exists():
        shutil.rmtree(args.delivery)
    args.delivery.mkdir(parents=True)
    manifest = []
    for r in good:
        slug, title = r["slug"], r["title"]
        src = args.out_dir / "selected" / f"{slug}.wav"
        dst = args.delivery / f"{slug}_YuE2.wav"
        shutil.copy2(src, dst)
        melody = r["melody"] or {}
        header = [
            f"# {title} · {slug} · 引擎 YuE2",
            f"> 音频：{dst.name}",
            f"> 模式：{modes.get(slug, '?')} / 时长 {r['duration']:.1f}s"
            f" / 动态 {r['arc'].get('verdict')} {fmt(r['arc'].get('lift_db'), 1, ' dB')}"
            f" / 歌词覆盖 {fmt(r['coverage'], 2)}",
            f"> 旋律：密度 {fmt(melody.get('density_ratio'))}"
            f" / 音区 +{fmt(melody.get('register_lift'))} 半音"
            f" / 钩子 {fmt(melody.get('hook_similarity'), 3)}"
            f" / 最高音 {fmt(melody.get('climax_position'))}",
            f"> Hook：{r['qa'].hook}",
            "",
        ]
        txt = "\n".join(header) + r["lyrics"].strip() + "\n"
        (args.delivery / f"{slug}_YuE2.txt").write_text(txt, encoding="utf-8")
        manifest.append({
            "file": dst.name,
            "lyrics_txt": f"{slug}_YuE2.txt",
            "slug": slug,
            "title": title,
            "mode": modes.get(slug),
            "hook": r["qa"].hook,
            "lyric_score": r["qa"].score,
            "duration_s": round(r["duration"], 2),
            "arc": {k: r["arc"].get(k) for k in
                    ("verdict", "lift_db", "peak_at", "body_db",
                     "mid_dip_db", "brightness_delta_hz")},
            "melody": {k: melody.get(k) for k in
                       ("passes", "score", "density_ratio", "register_lift",
                        "hook_similarity", "overall_range", "climax_position")},
            "coverage": r["coverage"],
            "selected": r["row"].get("selected"),
            "seed": r["row"].get("seed"),
        })

    (args.delivery / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 2026-10-26 中文精选",
        "",
        "只放入同时满足：词检通过、旋律门控通过、动态 build-up、起伏 ≥ "
        f"{args.min_lift:g} dB、歌词覆盖 ≥ {args.min_coverage:g}、时长 "
        f"{args.min_duration:.0f}–{args.max_duration:.0f}s 的曲目。",
        "",
        "| # | 曲目 | 模式 | 时长 | 起伏 | 旋律 | 覆盖 | hook |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for i, m in enumerate(manifest, 1):
        lines.append(
            f"| {i} | {m['title']} | {m['mode']} | {m['duration_s']:.0f}s | "
            f"{fmt(m['arc'].get('lift_db'), 1, ' dB')} | {m['melody'].get('score')} | "
            f"{fmt(m['coverage'], 2)} | {m['hook']} |")
    lines += ["", "按上表顺序听；WAV 与同名 TXT 一一对应。", ""]
    (args.delivery / "README.md").write_text("\n".join(lines), encoding="utf-8")

    if bad:
        ex = ["# 未入选", "", "| 曲目 | 原因 |", "|---|---|"]
        for r in bad:
            ex.append(f"| {r['title']} | {'；'.join(r['reasons'])} |")
        (args.delivery / "excluded.md").write_text("\n".join(ex) + "\n",
                                                   encoding="utf-8")

    print(f"delivery -> {args.delivery} ({len(manifest)} songs, "
          f"{len(bad)} excluded)")
    for m in manifest:
        print(f"  {m['title']} ({m['mode']}, {m['duration_s']:.0f}s, "
              f"melody {m['melody'].get('score')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
