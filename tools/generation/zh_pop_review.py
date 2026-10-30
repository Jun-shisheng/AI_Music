"""Render the Chinese-pilot listening sheet.

Reads the batch's own `index.json`, the lyric files and the per-take
`melody_qa.json`, then writes one Chinese `review.md` with:

  * a listening order (candidate A -> candidate B -> promoted take),
  * the lyric gate (hook, rhyme, concrete objects, warnings),
  * the symbolic melody numbers (chorus/verse density, register lift, hook
    similarity, climax position),
  * the audio numbers the batch already measured (arc, diction coverage),
  * a blank verdict table for the user.

It deliberately does not score "good"; the final column is the human's.

Usage:
  P=vendor/ACE-Step-1.5/.venv/Scripts/python.exe
  $P -I tools/generation/zh_pop_review.py \
      --out-dir outputs/2026-10-26_zh_pop \
      --inputs inputs/2026-10-26_zh_pop --mp3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "generation"))
from melody_qa import audit as melody_audit  # noqa: E402
from zh_lyric_qa import audit  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def fmt(value, suffix="", digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def pct(value, digits=0):
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def melody_for(out_dir: Path, take_stem: str) -> dict | None:
    """Recompute the melody gate from the saved plan, not the batch's stale row.

    The batch writes its melody numbers when the take finishes; if the gate's
    thresholds change during the run (2026-10-26: transposition-invariant hook
    similarity), the stored row no longer describes the same standard.
    """
    matches = list((out_dir / "candidates").glob(
        f"*/{take_stem}_artifacts/score.abc"))
    if not matches:
        return None
    try:
        report = melody_audit(matches[0].read_text(encoding="utf-8-sig"),
                              target="zh-pop")
        return asdict(report)
    except Exception:
        return None


def make_mp3(wav: Path, out: Path) -> bool:
    if not wav.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
         "-codec:a", "libmp3lame", "-q:a", "2", str(out)],
        capture_output=True, text=True)
    return proc.returncode == 0 and out.exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs" / "2026-10-26_zh_pop")
    ap.add_argument("--inputs", type=Path,
                    default=ROOT / "inputs" / "2026-10-26_zh_pop")
    ap.add_argument("--mp3", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    titles = load_json(args.inputs / "_titles.json", {})
    modes = load_json(args.inputs / "_modes.json", {})
    index = {row.get("title"): row for row in
             load_json(args.out_dir / "index.json", [])}

    lines = [
        "# 2026-10-26 中文曲小样听审",
        "",
        "> 4 首新词，每首 2 个候选；YuE2 原生 48 kHz，无 SVC、无 EQ、无压缩。",
        "> MP3 试听：`review/listen/<slug>_c1.mp3` / `_c2.mp3` / `_selected.mp3`。",
        "> 指标只负责拦住坏文件；**好听与否以你的听感为准**。",
        "",
        "## 试听顺序",
        "",
    ]
    for slug, title in titles.items():
        if slug not in index:
            continue
        cands = index[slug].get("candidates", [])
        if len(cands) >= 2:
            order = "先听 c1，再听 c2，最后听 selected"
        elif len(cands) == 1:
            order = "只跑了 c1（selected 就是它）"
        else:
            order = "听 selected"
        lines.append(f"1. **{title}**（{modes.get(slug, '?')}）— {order}")
    lines += ["", "## 指标总表", "",
              "| 曲目 | 模式 | hook | 词检 | 副歌/主歌密度 | 音区抬升 | 钩子相似度 | 最高音位置 | 起伏 | 歌词覆盖 |",
              "|---|---|---|---|---:|---:|---:|---:|---|---:|"]
    for slug, title in titles.items():
        row = index.get(slug)
        if not row:
            continue
        qa = audit((args.inputs / f"{slug}_lyrics.txt").read_text(encoding="utf-8"),
                   title=title, mode=modes.get(slug, "narrative"))
        melody = melody_for(args.out_dir, Path(row.get("selected", "")).stem) or row.get("melody") or {}
        arc = row.get("arc") or {}
        lines.append(
            f"| {title} | {modes.get(slug, '?')} | {qa.hook or '-'} | "
            f"{qa.score} {'通过' if qa.passes else '未过'} | "
            f"{fmt(melody.get('density_ratio'))} | "
            f"{fmt(melody.get('register_lift'), '半音')} | "
            f"{fmt(melody.get('hook_similarity'), digits=3)} | "
            f"{pct(melody.get('climax_position'))} | "
            f"{arc.get('verdict', '-')} / {fmt(arc.get('lift_db'), 'dB')} | "
            f"{fmt(row.get('diction'), digits=2)} |")

    lines += ["", "## 逐曲", ""]
    for slug, title in titles.items():
        row = index.get(slug)
        if not row:
            lines += [f"### {title}", "", "> 尚未出片。", ""]
            continue
        qa = audit((args.inputs / f"{slug}_lyrics.txt").read_text(encoding="utf-8"),
                   title=title, mode=modes.get(slug, "narrative"))
        melody = melody_for(args.out_dir, Path(row.get("selected", "")).stem) or row.get("melody") or {}
        arc = row.get("arc") or {}
        lines += [
            f"### {title}",
            "",
            f"- 模式：`{modes.get(slug)}`；hook：**{qa.hook or '-'}**（x{qa.hook_count}）",
            f"- 词检：{qa.score} 分；主韵 {qa.rhyme_main}，段内一致 "
            f"{qa.rhyme_consistency:.0%}；具体物件 {len(qa.concrete_hits)} 个；"
            f"三声连读 max {qa.tone3_runs_max}",
            f"- 旋律：副歌/主歌密度 {fmt(melody.get('density_ratio'))}，"
            f"音区 +{fmt(melody.get('register_lift'))} 半音，"
            f"钩子相似度 {fmt(melody.get('hook_similarity'), digits=3)}，"
            f"最高音在 {pct(melody.get('climax_position'))}",
            f"- 音频：{arc.get('verdict', '-')}，起伏 {fmt(arc.get('lift_db'), 'dB')}，"
            f"峰值位置 {pct(arc.get('peak_at'))}，"
            f"歌词覆盖 {fmt(row.get('diction'), digits=2)}",
            "",
        ]
        if qa.warnings:
            lines.append("词检提醒：" + "；".join(qa.warnings))
            lines.append("")
        if melody.get("passes") is False:
            lines.append("旋律门控未过：需要重跑或换候选。")
            lines.append("")
        lines.append("```")
        lines.append((args.inputs / f"{slug}_lyrics.txt").read_text(
            encoding="utf-8").strip())
        lines.append("```")
        lines.append("")
        lines.append("候选：")
        for cand in row.get("candidates", []):
            lines.append(
                f"- `{cand.get('file')}`：{cand.get('verdict')}，"
                f"lift {cand.get('lift_db')} dB，peak@{cand.get('peak_at')}，"
                f"coverage {cand.get('coverage')}，melody {cand.get('melody_score')}")
        lines.append("")
        lines.append("结论（请填）：保留 / 重跑 / 弃用：________")
        lines.append("")

        if args.mp3:
            listen = args.out_dir / "review" / "listen"
            selected = args.out_dir / "selected" / f"{slug}.wav"
            make_mp3(selected, listen / f"{slug}_selected.mp3")
            for cand in row.get("candidates", []):
                src = args.out_dir / "candidates" / cand["file"]
                # candidates live one directory down
                if not src.exists():
                    matches = list((args.out_dir / "candidates").glob(
                        f"*/{cand['file']}"))
                    src = matches[0] if matches else src
                stem = Path(cand["file"]).stem
                if stem.startswith(slug + "_"):
                    stem = stem[len(slug) + 1:]
                make_mp3(src, listen / f"{slug}_{stem}.mp3")

    lines += [
        "## 本轮问题",
        "",
        "1. 副歌是否比主歌更密、更高？",
        "2. hook 是否一耳朵能记住？有没有倒字/三声堆叠？",
        "3. 哪首值得进正式批次？哪首换 seed/换 hook 重跑？",
        "",
    ]
    out = args.out or (args.out_dir / "review.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"review -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
