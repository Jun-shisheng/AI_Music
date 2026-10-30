"""Finalize the extra 2026-10-20 fixed-voice covers into a dated delivery.

For each completed cover job this builds, on CPU only:
  * lyric QA for the converted vocals and for the source vocals (content proxy);
  * a 44.1 kHz listening mix from the untouched Demucs instrumental;
  * a bilingual (English + Chinese) lyrics sheet with the take's metrics;
  * a ~210 s delivery mix by looping the untouched instrumental at the ends,
    or the original-length mix when the body is already at/over the target;
  * manifest.json and a Chinese listening sheet for the delivery folder.

Nothing here grades timbre or musical taste, and nothing is pushed anywhere.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "tools/vocal_repair/.venv/Scripts/python.exe"
JOBS = ROOT / "data/voice_studio/jobs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/generation"))
import export_delivery as ed  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def qa(audio: Path, lyrics: Path, out_dir: Path) -> dict:
    cached = out_dir / "qa.json"
    if cached.is_file():
        return load(cached)
    command = [str(PY), str(ROOT / "tools/music_qa.py"), "--audio", str(audio),
               "--lyrics", str(lyrics), "--out-dir", str(out_dir), "--language", "en"]
    if subprocess.run(command, cwd=ROOT).returncode != 0:
        raise RuntimeError(f"QA failed for {audio}")
    return load(cached)


def cache_instrumental(log_path: Path, stem: str, source: Path | None = None) -> Path | None:
    """Locate the untouched Demucs instrumental for this source.

    A cache hit inside the cover job returns before writing `separation.log`,
    which is normal when the stems were pre-separated, so fall back to
    recomputing the exact Voice Studio cache key from the source audio.
    """
    if log_path.is_file():
        first = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        try:
            command = json.loads(first)
            cache = Path(command[command.index("-o") + 1])
            candidate = cache / "htdemucs" / stem / "no_vocals.wav"
            if candidate.is_file():
                return candidate
        except (ValueError, IndexError, json.JSONDecodeError):
            pass
    if source is not None and source.is_file():
        from tools.voice_studio import audio as v_audio
        from tools.voice_studio import workflow as v_workflow
        key = v_audio.fingerprint(dict(sha256=v_audio.digest(source), model="htdemucs",
                                       shifts=0, version=v_workflow.VERSION))
        candidate = (ROOT / "data/voice_studio/cache/separation" / key
                     / "htdemucs" / source.stem / "no_vocals.wav")
        if candidate.is_file():
            return candidate
    return None


def mix_44k(job: Path, stem: str, destination: Path, source: Path | None = None) -> bool:
    instrumental = cache_instrumental(job / "separation.log", stem, source)
    if instrumental is None:
        return False
    inst, sr = sf.read(instrumental, dtype="float64", always_2d=True)
    vocal, vr = sf.read(job / "converted_vocals.wav", dtype="float64")
    if vr != 24000 or sr != 44100:
        return False
    up = resample_poly(vocal, 147, 80)
    n = min(len(inst), len(up))
    if n <= 0:
        return False
    inst, up = inst[:n], up[:n]
    if up.ndim == 1:
        up = np.stack([up, up], axis=1)
    mix = inst + up
    peak = float(np.max(np.abs(mix)))
    if peak > 0.95:
        mix *= 0.95 / peak
    sf.write(destination, mix, sr, subtype="PCM_16")
    return True


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def extend(work: Path, mix: Path, instrumental: Path, out: Path, report: Path,
           target: float) -> dict:
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PY), str(ROOT / "tools/generation/extend_song_length.py"),
               "--mix", str(mix), "--instrumental", str(instrumental),
               "--out", str(out), "--target-seconds", str(target),
               "--report", str(report)]
    subprocess.run(command, cwd=ROOT, check=True)
    return load(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--local-out", type=Path, required=True)
    parser.add_argument("--delivery-out", type=Path, required=True)
    parser.add_argument("--target-seconds", type=float, default=210.0)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    entries = json.loads(args.list.read_text(encoding="utf-8"))
    if args.only:
        entries = [item for item in entries if item["slug"] in set(args.only)]
    args.local_out.mkdir(parents=True, exist_ok=True)
    args.delivery_out.mkdir(parents=True, exist_ok=True)
    qa_root = args.work_dir / "qa"

    records = []
    for entry in entries:
        slug = entry["slug"]
        job = JOBS / f"cover_{slug}_full_v1"
        manifest_path = job / "manifest.json"
        if not manifest_path.is_file():
            print(f"SKIP {slug}: no manifest", flush=True)
            continue
        manifest = load(manifest_path)
        if manifest.get("status") not in {"awaiting_listening", "technical_review_required"}:
            print(f"SKIP {slug}: status {manifest.get('status')}", flush=True)
            continue
        outputs = manifest.get("outputs", {})
        if "mix" not in outputs or "vocals" not in outputs:
            print(f"SKIP {slug}: outputs incomplete", flush=True)
            continue
        lyrics_dir = ROOT / entry["lyrics_dir"]
        lyrics_en = lyrics_dir / f"{slug}_lyrics.txt"
        lyrics_zh = lyrics_dir / f"{slug}_lyrics_zh.txt"
        if not lyrics_en.is_file():
            print(f"SKIP {slug}: no lyrics", flush=True)
            continue

        with ThreadPoolExecutor(max_workers=2) as pool:
            cover_future = pool.submit(qa, job / "converted_vocals.wav", lyrics_en,
                                       qa_root / f"cover_{slug}_full_v1")
            source_future = pool.submit(qa, job / "target_vocals.wav", lyrics_en,
                                        qa_root / f"cover_{slug}_full_v1_source")
            cover_qa, source_qa = cover_future.result(), source_future.result()
        c, s = cover_qa["comparison"], source_qa["comparison"]

        native_mix = args.local_out / f"{slug}_full_fixed_voice.wav"
        native_vocals = args.local_out / f"{slug}_full_fixed_voice_vocals.wav"
        mix44 = args.local_out / f"{slug}_full_fixed_voice_mix_44k.wav"
        shutil.copy2(job / "review_mix.wav", native_mix)
        shutil.copy2(job / "converted_vocals.wav", native_vocals)
        source_path = Path(manifest["inputs"]["source"])
        made44 = mix_44k(job, source_path.stem, mix44, source_path)
        body_seconds = duration(mix44 if made44 else native_mix)
        instrument = cache_instrumental(job / "separation.log", source_path.stem,
                                        source_path)
        take = "full_v1"
        technical = manifest.get("technical_qa", {})
        body, warnings = ed.bilingual_body(lyrics_en.read_text(encoding="utf-8"),
                                           lyrics_zh.read_text(encoding="utf-8"))
        notes = [w for w in warnings if "no Chinese gloss" not in w]
        title = entry.get("title_zh") or slug
        base = mix44 if made44 else native_mix

        # Original-length A/B copy: same take, no added intro/outro.
        orig_stem = f"{slug}_SVC_orig"
        shutil.copy2(base, args.delivery_out / f"{orig_stem}.wav")
        orig_header = [
            f"# {title} · {slug} · 引擎 SVC（固定女声 iron_rain_female_v1）· 原长版",
            f"# 对应音频 {orig_stem}.wav",
            "# 与 _SVC_3m30 同一 take；未追加前奏/尾奏，长度由原曲决定。",
            "# 说明：SVC 人声为 24 kHz；44.1 kHz 混音只恢复伴奏高频。",
        ]
        if notes:
            orig_header.append("# 歌词对齐提示：" + "；".join(notes))
        (args.delivery_out / f"{orig_stem}.txt").write_text(
            "\n".join(orig_header) + "\n\n" + body, encoding="utf-8")

        render = None
        if made44 and instrument is not None and body_seconds + 0.5 < args.target_seconds:
            stem = f"{slug}_SVC_3m30"
            delivery_wav = args.delivery_out / f"{stem}.wav"
            render = extend(args.work_dir / "render", mix44, instrument, delivery_wav,
                            args.work_dir / "render" / f"{slug}_render.json",
                            args.target_seconds)
        else:
            stem = orig_stem
            delivery_wav = args.delivery_out / f"{stem}.wav"
        extended = render is not None

        header = [
            f"# {title} · {slug} · 引擎 SVC（固定女声 iron_rain_female_v1）· "
            + ("3分30秒版" if extended else "原长版"),
            f"# 对应音频 {stem}.wav",
            f"# 取用 take {take}；ASR 代理 {c['normalized_character_similarity']:.3f}"
            f"（{c['approximate_lines_recovered']}/{c['lines_total']} 句，整曲）。",
            "# 人声与歌曲主体未改动；SVC 人声为 24 kHz，44.1 kHz 混音只恢复伴奏高频。",
        ]
        if extended:
            header.append(
                "# 仅用未改动伴奏追加器乐前奏/尾奏到约 210 秒（300 ms 交叉淡化，尾奏 4 秒淡出）。")
        else:
            header.append(
                f"# 原曲主体 {body_seconds:.1f} 秒，已接近/超过 3 分 30 秒，因此不追加尾奏。")
        header.append(
            f"# 原唱干声同口径：{s['normalized_character_similarity']:.3f}"
            f"（{s['approximate_lines_recovered']}/{s['lines_total']} 句）。")
        if notes:
            header.append("# 歌词对齐提示：" + "；".join(notes))
        header.append("# 说明：自动转写只用于定位问题，不是音色或审美评分。")
        txt = args.delivery_out / f"{stem}.txt"
        txt.write_text("\n".join(header) + "\n\n" + body, encoding="utf-8")

        record = {
            "file": delivery_wav.name,
            "engine": "SVC",
            "title": slug,
            "chinese_title": title,
            "take": take,
            "duration_seconds": round(duration(delivery_wav), 3),
            "body_seconds": round(body_seconds, 3),
            "intro_seconds": (render or {}).get("intro_seconds", 0.0),
            "outro_seconds": (render or {}).get("outro_seconds", 0.0),
            "cover_similarity": round(c["normalized_character_similarity"], 3),
            "cover_lines": f"{c['approximate_lines_recovered']}/{c['lines_total']}",
            "source_similarity": round(s["normalized_character_similarity"], 3),
            "source_lines": f"{s['approximate_lines_recovered']}/{s['lines_total']}",
            "dropout_ratio": technical.get("dropout_ratio"),
            "clipping_ratio": technical.get("clipping_ratio"),
            "origin_batch": entry.get("origin"),
            "metric_region": "full track",
            "lyrics_txt": f"{args.delivery_out.name}/{txt.name}",
        }
        records.append(record)
        print(f"DONE {slug} -> {delivery_wav.name} ({record['duration_seconds']}s)", flush=True)

    write_json(args.delivery_out / "manifest.json", records)

    lines = [f"固定女声新增曲目（2026-10-20）· {len(records)} 首", ""]
    lines.append("| # | 曲目 | 原长 | 交付 | ASR 代理 | 原唱同口径 |")
    lines.append("|---|---|---|---|---|---|")
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
        "",
        "听审重点：音色身份是否一致、咬字、12 秒分句接缝、尾奏循环是否自然。",
        "ASR 只是定位工具，不是音色或审美评分；人声高频上限 12 kHz 来自 SVC 后端。",
    ]
    (args.delivery_out / f"{args.delivery_out.name}_review.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
