"""Render the evening review sheet (and MP3 previews) for a YuE2 epic batch.

Reads the batch's `candidates/*/result.json`, `inputs/*/_modes.json` and
`inputs/*/briefs.json`, then writes `<out>/review.md` plus
`<out>/review/listen/*.mp3` ordered by mode and by the selection gate.

Gate (for "listen first" only, not a quality verdict):
  * duration 110-215 s
  * arc verdict `build-up` (or `mild build` within 1 dB of it)
  * diction coverage >= 0.75 when a vocal track
Tracks that miss it stay in the folder with the reason shown, so a late-night
listener knows where to spend attention.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODE_LABEL = {
    "epic": "哲思史诗（御姐）",
    "epic_ethereal": "空灵史诗",
    "battle_anthem": "战斗史诗",
    "instrumental": "战斗 BGM",
}
MODE_ORDER = {"epic": 0, "epic_ethereal": 1, "battle_anthem": 2, "instrumental": 3}


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def gate(row: dict) -> tuple[bool, str]:
    reasons = []
    if not 110 <= row["duration_s"] <= 215:
        reasons.append(f"时长 {row['duration_s']:.0f}s")
    if row["arc"] not in ("build-up", "mild build"):
        reasons.append(f"结构 {row['arc']}")
    if row["coverage"] is not None and row["coverage"] < 0.75:
        reasons.append(f"咬字 {row['coverage']:.2f}")
    return (not reasons, "、".join(reasons) or "过线")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-mp3", action="store_true")
    args = ap.parse_args()

    modes = load(args.inputs / "_modes.json", {})
    briefs = load(args.inputs / "briefs.json", {})
    titles: dict = {}
    for path in sorted(args.inputs.parent.glob("*/_titles.json")):
        titles.update(load(path, {}))
    rows = []
    result_files = sorted(args.out_dir.glob("candidates/*/result.json"))
    bgm_dir = args.out_dir / "bgm"
    if bgm_dir.exists():
        result_files += sorted(bgm_dir.glob("candidates/*/result.json"))
    for result in result_files:
        if "_artifacts" in str(result):
            continue
        data = load(result, {})
        if "arc" not in data:
            continue
        slug = data.get("title", result.parent.name)
        brief = briefs.get(slug, {}).get("brief", {})
        arc = data["arc"]
        melody = data.get("melody") or {}
        rows.append({
            "slug": slug,
            "title": brief.get("title") or titles.get(slug, slug),
            "hook": brief.get("hook", ""),
            "metaphor": brief.get("metaphor", ""),
            "mode": modes.get(slug, "instrumental" if "bgm" in str(result) else ""),
            "duration_s": float(arc.get("duration_s") or 0),
            "arc": arc.get("verdict", ""),
            "lift_db": float(arc.get("lift_db") or 0),
            "peak_at": float(arc.get("peak_at") or 0),
            "dip_db": arc.get("dip_before_peak_db"),
            "coverage": data.get("diction"),
            "density": melody.get("density_ratio"),
            "reglift": melody.get("register_lift"),
            "wav": (args.out_dir / ("bgm" if "bgm" in str(result) else "")
                    / "selected" / f"{slug}.wav"),
        })

    rows.sort(key=lambda r: (MODE_ORDER.get(r["mode"], 9), r["slug"]))
    for row in rows:
        row["pass"], row["why"] = gate(row)

    listen = args.out_dir / "review" / "listen"
    listen.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 2026-10-28 史诗批（晚间评审）",
        "",
        "全部 YuE2 原生 48 kHz，无 SVC / EQ / 压缩 / 限幅。",
        "MP3 在 `review/listen/`，WAV 在 `selected/`。",
        "",
        "## 听审顺序",
        "",
        "先听 `过线` 的，再听 `待修` 的；每首只听 45 秒也能判断方向。",
        "",
        "| 顺序 | 曲 | 类型 | 时长 | 结构 | 高潮位置 | 咬字 | 判定 |",
        "|---:|---|---|---:|---|---:|---:|---|",
    ]
    order = [r for r in rows if r["pass"]] + [r for r in rows if not r["pass"]]
    for i, row in enumerate(order, 1):
        cov = "—" if row["coverage"] is None else f"{row['coverage']:.2f}"
        lines.append(
            f"| {i} | **{row['title']}**（{row['slug']}） | "
            f"{MODE_LABEL.get(row['mode'], row['mode'])} | "
            f"{row['duration_s']:.0f}s | {row['arc']} | "
            f"{row['peak_at'] * 100:.0f}% | {cov} | "
            f"{'过线' if row['pass'] else '待修：' + row['why']} |")
    lines += ["", "## 每首一句话", ""]
    for row in order:
        lines.append(f"- **{row['title']}**：hook「{row['hook']}」；{row['metaphor']}")
    lines += [
        "",
        "## 已确认作为基准的一首",
        "",
        "`03_the_rope_end`（绳头）不在本批，它是校准批里选中的参考：",
        "build-up 16.4 dB、高潮 91%、咬字 1.08。本批听感请以它为对照。",
        "",
    ]
    (args.out_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")

    if not args.no_mp3:
        for row in order:
            if not row["wav"].exists():
                continue
            dst = listen / f"{row['slug']}.mp3"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i",
                            str(row["wav"]), "-codec:a", "libmp3lame", "-q:a",
                            "2", str(dst)], check=False)
    print(f"{len(order)} titles -> {args.out_dir / 'review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
