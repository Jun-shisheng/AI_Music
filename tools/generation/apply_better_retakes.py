"""Replace delivery takes with v2 only when the content proxy clearly improves.

The v2 job is the same fixed singer with seed 43 / 48 steps. For every slug the
v2 vocals and the source vocals are scored against the same lyrics; a v2 take
replaces v1 only when its normalized character similarity beats v1 by at least
`--margin`. The delivery WAVs, TXT headers, manifest and review sheet are then
rebuilt from the v2 takes; otherwise the v1 files stay untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/generation"))
import finalize_more_covers as fin  # noqa: E402
import export_delivery as ed  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def review_text(records: list[dict]) -> str:
    lines = [f"固定女声新增曲目（2026-10-20）· {len(records)} 首", "",
             "| # | 曲目 | 原长 | 交付 | ASR 代理 | 原唱同口径 |",
             "|---|---|---|---|---|---|"]
    for index, item in enumerate(records, 1):
        lines.append(
            f"| {index:02d} | {item['chinese_title']}（{item['title']}） | "
            f"{item['body_seconds']:.1f} | {item['duration_seconds']:.1f} | "
            f"{item['cover_similarity']:.3f}（{item['cover_lines']}） | "
            f"{item['source_similarity']:.3f}（{item['source_lines']}） |")
    lines += [
        "",
        "全部为同一固定女声 SoulX-Singer SVC 翻唱：Demucs 分轨后用原曲歌词与旋律换声，",
        "常数增益电平匹配后与未改动伴奏直接相加，没有 EQ、降噪、压缩或限幅。",
        "约 3 分 30 秒版只在两端循环未改动伴奏，人声与歌曲主体不动；",
        "主体本身已接近 210 秒的曲目交付原长版。",
        "每首另附 `_SVC_orig.wav`（同一 take，不追加前奏/尾奏）供直接对照。",
        "标记为 v2 的曲目使用 seed 43 / 48 steps 重跑，其余为 seed 42 / 32 steps。",
        "",
        "听审重点：音色身份是否一致、咬字、12 秒分句接缝、尾奏循环是否自然。",
        "ASR 只是定位工具，不是音色或审美评分；人声高频上限 12 kHz 来自 SVC 后端。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--local-out", type=Path, required=True)
    ap.add_argument("--delivery-out", type=Path, required=True)
    ap.add_argument("--slugs", nargs="+", required=True)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--target-seconds", type=float, default=210.0)
    args = ap.parse_args()

    entries = {e["slug"]: e for e in load(args.list)}
    qa_root = args.work_dir / "qa"
    manifest_path = args.delivery_out / "manifest.json"
    records = load(manifest_path)
    by_title = {r["title"]: r for r in records}
    decisions = []

    for slug in args.slugs:
        entry = entries[slug]
        v2_job = ROOT / "data/voice_studio/jobs" / f"cover_{slug}_full_v2"
        if not (v2_job / "manifest.json").is_file():
            decisions.append((slug, "skip", None, None))
            continue
        v2_manifest = load(v2_job / "manifest.json")
        if v2_manifest.get("status") not in {"awaiting_listening", "technical_review_required"}:
            decisions.append((slug, f"skip-status-{v2_manifest.get('status')}", None, None))
            continue
        lyrics_dir = ROOT / entry["lyrics_dir"]
        lyrics_en = lyrics_dir / f"{slug}_lyrics.txt"
        lyrics_zh = lyrics_dir / f"{slug}_lyrics_zh.txt"
        v2_qa = fin.qa(v2_job / "converted_vocals.wav", lyrics_en,
                       qa_root / f"cover_{slug}_full_v2")
        v1_qa_path = qa_root / f"cover_{slug}_full_v1" / "qa.json"
        if not v1_qa_path.is_file():
            decisions.append((slug, "skip-no-v1-qa", None, None))
            continue
        v1_sim = load(v1_qa_path)["comparison"]["normalized_character_similarity"]
        v2_sim = v2_qa["comparison"]["normalized_character_similarity"]
        technical = v2_manifest.get("technical_qa") or {}
        dropout = technical.get("dropout_ratio") or 0.0
        clipping = technical.get("clipping_ratio") or 0.0
        if dropout > 0.02 or clipping > 0.001:
            decisions.append((slug, f"keep-v1 (v2 technical dropout={dropout} clipping={clipping})",
                              v1_sim, v2_sim))
            continue
        if v2_sim < v1_sim + args.margin:
            decisions.append((slug, f"keep-v1 ({v2_sim:.3f} <= {v1_sim:.3f})", v1_sim, v2_sim))
            continue

        source = Path(v2_manifest["inputs"]["source"])
        mix44 = args.local_out / f"{slug}_full_v2_fixed_voice_mix_44k.wav"
        made44 = fin.mix_44k(v2_job, source.stem, mix44, source)
        if not made44:
            decisions.append((slug, "skip-no-44k", v1_sim, v2_sim))
            continue
        body_seconds = fin.duration(mix44)
        instrument = fin.cache_instrumental(v2_job / "separation.log", source.stem, source)
        orig_stem = f"{slug}_SVC_orig"
        shutil.copy2(mix44, args.delivery_out / f"{orig_stem}.wav")
        main_stem = orig_stem
        render = None
        if instrument is not None and body_seconds + 0.5 < args.target_seconds:
            main_stem = f"{slug}_SVC_3m30"
            render = fin.extend(args.work_dir / "render", mix44, instrument,
                                args.delivery_out / f"{main_stem}.wav",
                                args.work_dir / "render" / f"{slug}_v2_render.json",
                                args.target_seconds)
        else:
            shutil.copy2(mix44, args.delivery_out / f"{main_stem}.wav")

        c = v2_qa["comparison"]
        source_qa_path = qa_root / f"cover_{slug}_full_v1_source" / "qa.json"
        s = load(source_qa_path)["comparison"] if source_qa_path.is_file() else {
            "normalized_character_similarity": float("nan"),
            "approximate_lines_recovered": 0, "lines_total": 0}
        body, warnings = ed.bilingual_body(lyrics_en.read_text(encoding="utf-8"),
                                           lyrics_zh.read_text(encoding="utf-8"))
        notes = [w for w in warnings if "no Chinese gloss" not in w]
        title = entry.get("title_zh") or slug
        extended = render is not None

        orig_header = [
            f"# {title} · {slug} · 引擎 SVC（固定女声 iron_rain_female_v1）· 原长版",
            f"# 对应音频 {orig_stem}.wav",
            "# 与 _SVC_3m30 同一 take；未追加前奏/尾奏，长度由原曲决定。",
            "# 取用 take full_v2（seed 43 / 48 steps，内容代理优于 full_v1）。",
            "# 说明：SVC 人声为 24 kHz；44.1 kHz 混音只恢复伴奏高频。",
        ]
        if notes:
            orig_header.append("# 歌词对齐提示：" + "；".join(notes))
        (args.delivery_out / f"{orig_stem}.txt").write_text(
            "\n".join(orig_header) + "\n\n" + body, encoding="utf-8")

        if extended:
            header = [
                f"# {title} · {slug} · 引擎 SVC（固定女声 iron_rain_female_v1）· 3分30秒版",
                f"# 对应音频 {main_stem}.wav",
                f"# 取用 take full_v2（seed 43 / 48 steps）；ASR 代理 "
                f"{c['normalized_character_similarity']:.3f}"
                f"（{c['approximate_lines_recovered']}/{c['lines_total']} 句，整曲）。",
                "# 人声与歌曲主体未改动；SVC 人声为 24 kHz，44.1 kHz 混音只恢复伴奏高频。",
                "# 仅用未改动伴奏追加器乐前奏/尾奏到约 210 秒（300 ms 交叉淡化，尾奏 4 秒淡出）。",
                f"# 原唱干声同口径：{s['normalized_character_similarity']:.3f}"
                f"（{s['approximate_lines_recovered']}/{s['lines_total']} 句）。",
                "# 说明：自动转写只用于定位问题，不是音色或审美评分。",
            ]
            if notes:
                header.append("# 歌词对齐提示：" + "；".join(notes))
            (args.delivery_out / f"{main_stem}.txt").write_text(
                "\n".join(header) + "\n\n" + body, encoding="utf-8")

        record = by_title[slug]
        record.update({
            "file": f"{main_stem}.wav",
            "take": "full_v2",
            "duration_seconds": round(fin.duration(args.delivery_out / f"{main_stem}.wav"), 3),
            "body_seconds": round(body_seconds, 3),
            "intro_seconds": (render or {}).get("intro_seconds", 0.0),
            "outro_seconds": (render or {}).get("outro_seconds", 0.0),
            "cover_similarity": round(v2_sim, 3),
            "cover_lines": f"{c['approximate_lines_recovered']}/{c['lines_total']}",
            "source_similarity": round(s["normalized_character_similarity"], 3),
            "source_lines": f"{s['approximate_lines_recovered']}/{s['lines_total']}",
            "lyrics_txt": f"{args.delivery_out.name}/{main_stem}.txt",
        })
        decisions.append((slug, f"replaced-v1 ({v2_sim:.3f} > {v1_sim:.3f})", v1_sim, v2_sim))

    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    (args.delivery_out / f"{args.delivery_out.name}_review.md").write_text(
        review_text(records), encoding="utf-8")
    for slug, decision, v1, v2 in decisions:
        print(f"{slug}: {decision}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
