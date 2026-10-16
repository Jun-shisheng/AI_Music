"""Render a listening-review sheet for one or more battle batches.

Reads `index.json` from each `--batch-dir` (written by battle_batch.py or
acestep_batch_battle.py) plus the slug -> 中文标题 / language maps from the
matching `inputs/<batch>` folder, and writes `review.md`: ranked selections, the
dynamic-arc picture for every promoted take, whether the climax actually gets
bigger, and the full candidate table so the human can see what was rejected.

Written as UTF-8 by design -- the console here is GBK and mangles CJK.

Usage:
  python tools/generation/battle_review.py \
      --batch-dir outputs/battle_batch outputs/battle_batch2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import arc_qa  # noqa: E402
from export_delivery import brief_folders  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

LANG_LABEL = {"zh": "中文", "en": "英文", "auto": "中英混搭"}


def engine_of(batch: Path) -> str:
    """Which backend produced a batch dir (the folder name is the only marker)."""
    name = batch.name.lower()
    if "yue2" in name:
        return "YuE2"
    if "ace" in name:
        return "ACE-Step"
    return "HeartMuLa"


def load_json(path: Path, skip_comment: bool = True) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items()
            if not (skip_comment and k.startswith("_"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", type=Path, nargs="+",
                    default=[PROJECT_ROOT / "outputs" / "battle_batch"])
    ap.add_argument("--engine", default=None,
                    help="engine label for every batch dir, overriding the "
                         "folder-name heuristic (a dated batch dir carries no "
                         "engine name)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Slug -> title / language are merged across every brief folder, because one
    # output batch (e.g. the YuE2 pass) can draw titles from several input
    # folders. Slugs are unique, so merging is safe and avoids every caller
    # having to name the right inputs dir. Discovery is by content, not by the
    # `battle_batch*` prefix, so a dated batch folder works too.
    all_titles: dict[str, str] = {}
    all_langs: dict[str, str] = {}
    for folder in brief_folders():
        all_titles.update(load_json(folder / "_titles.json"))
        all_langs.update(load_json(folder / "_langs.json"))

    entries: list[tuple[dict, Path, dict, dict]] = []   # row, dir, titles, langs
    for batch in args.batch_dir:
        # Aggregate the per-title result.json files rather than index.json: a
        # batch can be produced by several runs (the OOM fill-ins run one title
        # per invocation) and each run overwrites index.json with its own rows.
        # result.json is written per promoted title, so it is the durable record.
        rows_by_title: dict[str, tuple[float, dict]] = {}
        for result in batch.glob("candidates/*/result.json"):
            try:
                row = json.loads(result.read_text(encoding="utf-8"))
            except Exception:
                continue
            title = row.get("title")
            if not title:
                continue
            # A result.json can outlive its take: the length gate discards an
            # out-of-range promote and (when the wall-clock budget runs out)
            # never replaces it. Listing such a title would put a file name in
            # the sheet that is not there to listen to.
            if not (batch / "selected" / f"{title}.wav").exists():
                continue
            mtime = result.stat().st_mtime
            if title not in rows_by_title or mtime > rows_by_title[title][0]:
                rows_by_title[title] = (mtime, row)
        if not rows_by_title:
            index_path = batch / "index.json"
            if not index_path.exists():
                print(f"[skip] {batch}: no results yet")
                continue
            for row in json.loads(index_path.read_text(encoding="utf-8")):
                rows_by_title[row.get("title", "?")] = (0.0, row)
        for _, row in sorted(rows_by_title.values(), key=lambda x: x[1].get("title", "")):
            entries.append((row, batch, all_titles, all_langs))

    if not entries:
        print("nothing to review")
        return 1

    out = args.out or (args.batch_dir[0] / "review.md")
    lines: list[str] = ["# 战斗曲试听清单 (battle batch)\n"]
    lines.append(
        "- 候选规则：每首多个候选，按「结构弧线(欲扬先抑) + 对比度/节奏 + 歌词还原度」"
        "自动排名\n"
        "- 交付文件：`selected/<slug>.wav`（线性响度归一化）；原件："
        "`selected/<slug>_raw.wav`（零处理，保留）\n"
        "- 弧线图例：左弱 ` ` → 右强 `@`（每格约 2 秒）\n")

    lines.append("\n## 选中一览\n")
    lines.append("| 引擎 | 曲目 | 语言 | 结构 | 起伏(dB) | 高潮位置 | 首句(s) | "
                 "躯干(dB) | 桥段回落 | 节奏 | 歌词还原 | 文件 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row, batch, titles, langs in entries:
        arc = row["arc"]
        cn = titles.get(row["title"], row["title"])
        lang = LANG_LABEL.get(langs.get(row["title"], "zh"), langs.get(row["title"], "中文"))
        onset = row.get("first_vocal_s")
        lines.append(
            f"| {args.engine or engine_of(batch)} | {cn} `{row['title']}` | {lang} | {arc['verdict']} "
            f"| {arc['lift_db']:+.1f} | {arc['peak_at']:.2f} "
            f"| {f'{onset:.1f}' if onset is not None else '-'} "
            f"| {arc.get('body_db', 0):+.1f} "
            f"| {arc.get('dip_before_peak_db', 0):+.1f} "
            f"| {arc.get('onset_mean', 0):.3f} "
            f"| {row['diction'] if row['diction'] is not None else '-'} "
            f"| `{row['delivery']}` |")

    lines.append("\n## 弧线（每条 = 该曲选中版本的能量走向）\n")
    lines.append("```")
    for row, batch, titles, langs in entries:
        cn = titles.get(row["title"], row["title"])
        lines.append(f"{cn:<10} |{row['envelope']}|  {row['arc']['verdict']}")
    lines.append("```\n")

    # A real tutti climax adds brass / cymbals / choir (the spectral centre
    # rises); a climax that merely rides the gain leaves it flat. Computed here
    # rather than in the batch so it also covers runs made before the metric
    # existed.
    lines.append("\n## 高潮是否「变大」而不只是「变响」\n")
    lines.append("| 曲目 | 开头质心 Hz | 高潮质心 Hz | 变化 | 判读 |")
    lines.append("|---|---|---|---|---|")
    for row, batch, titles, langs in entries:
        cn = titles.get(row["title"], row["title"])
        path = batch / "selected" / row["delivery"]
        if not path.exists():
            continue
        try:
            b = arc_qa.brightness(path, row["arc"]["peak_at"])
        except Exception as exc:  # never let a metric break the review
            lines.append(f"| {cn} | - | - | - | {exc!r} |")
            continue
        delta = b["brightness_delta_hz"]
        read = ("高潮更亮更厚（加分）" if delta > 150 else
                "基本持平" if abs(delta) <= 150 else
                "高潮反而更暗（偏薄/被压暗）")
        lines.append(f"| {cn} | {b['centroid_intro_hz']} | {b['centroid_peak_hz']} "
                     f"| {delta:+d} Hz | {read} |")
    lines.append("")

    lines.append("\n## 各曲候选明细\n")
    for row, batch, titles, langs in entries:
        cn = titles.get(row["title"], "")
        lines.append(f"\n### {cn} `{row['title']}`\n")
        lines.append(f"- 选中：`{row['selected']}`（seed {row['seed']}），"
                     f"结构 `{row['arc']['verdict']}`，"
                     f"起伏 {row['arc']['lift_db']:+.1f} dB，"
                     f"高潮在 {row['arc']['peak_at']:.2f} 处，"
                     f"高潮前最低点 {row['arc'].get('dip_before_peak_db', 0):+.1f} dB，"
                     f"安静段占比 {row['arc']['quiet_share']:.2f}")
        lines.append(f"- 原件/交付：`{row['raw']}` / `{row['delivery']}`"
                     f"（归一化增益 {row['gain_db']} dB）")
        lines.append("\n| 候选 | seed | 结构 | 起伏 dB | 高潮位置 | 歌词还原 | 得分 |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in row["candidates"]:
            mark = " ✅" if c["file"] == row["selected"] else ""
            cov = c["coverage"] if c["coverage"] is not None else "-"
            lines.append(f"| `{c['file']}`{mark} | {c['seed']} | {c['verdict']} "
                         f"| {c['lift_db']:+.1f} | {c['peak_at']:.2f} | {cov} | {c['score']} |")

    lines.append("\n## 怎么读这份清单\n")
    lines.append(
        "- **指标**：`intro_db` 开头电平（越负＝越收）；`lift_db` 全曲起伏；"
        "`peak_at` 高潮位置（越接近 1 越晚）；`dip_before_peak_db` 高潮前的最低点"
        "（越负＝桥段回落越明显）。要的是「开头收、桥段落、高潮晚而大」。\n"
        "- **首句(s)**：whisper 词级时间戳量到的第一句人声落点"
        "（`first_vocal_s`）。战斗曲类的目标是不超过 20 秒，"
        "实际由前奏裁剪直接压到几秒。\n"
        "- **躯干(dB)**：`body_db`，全曲中位电平相对自身峰值的高度。越负＝"
        "整首曲子里真正“低”的部分越多；接近 0 意味着一直贴着峰值响，"
        "听感就是“平”。\n"
        "- **节奏**：`onset_mean`，归一化谱通量（打击感/律动密度），"
        "越大越有节奏推动力。\n"
        "- **歌词还原 coverage** 是 whisper 反查比例，作旁证；中英混搭曲用 "
        "`auto` 检测，数值偏差更大，务必以耳朵为准。\n")

    lines.append("\n## 请重点确认\n")
    lines.append(
        "1. **前奏**：首句是否在几秒内出现（参考曲通常在 0:02-0:05 开口）。\n"
        "2. **起伏**：低的地方够不够低、高潮够不够高（看 `躯干(dB)`、"
        "`桥段回落` 与弧线图）。\n"
        "3. **重复**：副歌钩子的重复是否像“同一个钩子回来了”，"
        "而不是每次换一段旋律（开头/末副歌/尾奏应尽量是新词）。\n"
        "4. 顺带确认：人声音色/唱法、管弦+电子配比、燃点位置、吐字清楚度。\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(entries)} titles from {len(args.batch_dir)} batch dirs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
