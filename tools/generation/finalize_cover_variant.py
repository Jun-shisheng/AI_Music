"""Finalize an alternative SVC take (e.g. full_v2) into the dated folder.

Same output contract as finalize_full_covers.py, but the filenames carry the
variant so the first pass stays intact for A/B. The CPU lyric QA is reused when
already present; the source-vocal baseline is the full_v1 QA if available.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "tools/vocal_repair/.venv/Scripts/python.exe"
JOBS = ROOT / "data/voice_studio/jobs"
OUT = ROOT / "outputs/2026-10-20/_work/finalize"
FINISHED = OUT / "finished"
INPUTS = ROOT / "inputs/2026-10-15_epic_en"

sys.path.insert(0, str(ROOT / "tools/generation"))
import export_delivery as ed  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def qa(audio: Path, lyrics: Path, out_dir: Path) -> dict:
    if (out_dir / "qa.json").is_file():
        return load(out_dir / "qa.json")
    command = [str(PY), str(ROOT / "tools/music_qa.py"), "--audio", str(audio),
               "--lyrics", str(lyrics), "--out-dir", str(out_dir), "--language", "en"]
    if subprocess.run(command, cwd=ROOT).returncode != 0:
        raise RuntimeError(f"QA failed for {audio}")
    return load(out_dir / "qa.json")


def cache_instrumental(log_path: Path, stem: str) -> Path | None:
    if not log_path.is_file():
        return None
    first = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    try:
        command = json.loads(first)
        cache = Path(command[command.index("-o") + 1])
    except (ValueError, IndexError, json.JSONDecodeError):
        return None
    candidate = cache / "htdemucs" / stem / "no_vocals.wav"
    return candidate if candidate.is_file() else None


def mix_44k(job: Path, stem: str, destination: Path) -> bool:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    instrumental = cache_instrumental(job / "separation.log", stem)
    if instrumental is None:
        return False
    inst, sr = sf.read(instrumental, dtype="float64", always_2d=True)
    vocal, vr = sf.read(job / "converted_vocals.wav", dtype="float64")
    if vr != 24000 or sr != 44100:
        return False
    up = resample_poly(vocal, 147, 80)
    n = min(len(inst), len(up))
    inst, up = inst[:n], up[:n]
    if up.ndim == 1:
        up = np.stack([up, up], axis=1)
    mix = inst + up
    peak = float(np.max(np.abs(mix)))
    if peak > 0.95:
        mix *= 0.95 / peak
    sf.write(destination, mix, sr, subtype="PCM_16")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slugs", nargs="+", required=True)
    parser.add_argument("--variant", default="full_v2")
    args = parser.parse_args()
    FINISHED.mkdir(parents=True, exist_ok=True)
    titles = load(INPUTS / "_titles.json")
    for slug in args.slugs:
        job = JOBS / f"cover_{slug}_{args.variant}"
        if not (job / "manifest.json").is_file():
            print(f"SKIP {slug}: no job")
            continue
        manifest = load(job / "manifest.json")
        if manifest.get("status") not in {"awaiting_listening", "technical_review_required"}:
            print(f"SKIP {slug}: {manifest.get('status')}")
            continue
        lyrics_en = INPUTS / f"{slug}_lyrics.txt"
        lyrics_zh = INPUTS / f"{slug}_lyrics_zh.txt"
        cover_qa = qa(job / "converted_vocals.wav", lyrics_en, OUT / "qa" / f"cover_{slug}_{args.variant}")
        source_qa_path = OUT / "qa" / f"cover_{slug}_full_v1_source" / "qa.json"
        source = load(source_qa_path) if source_qa_path.is_file() else None
        c = cover_qa["comparison"]
        technical = manifest.get("technical_qa", {})
        body, warnings = ed.bilingual_body(lyrics_en.read_text(encoding="utf-8"),
                                           lyrics_zh.read_text(encoding="utf-8"))
        title = titles.get(slug, slug)
        header = [
            f"# {title} · {slug} · 固定女声全曲 {args.variant}（替代 take）",
            f"# 引擎：SoulX-Singer SVC；歌手档案 iron_rain_female_v1；seed {manifest['inputs'].get('seed')}，steps {manifest['inputs'].get('steps')}",
            f"# 音频：{slug}_{args.variant}_fixed_voice.wav / _mix_44k.wav / _vocals.wav",
            f"# 时长 {technical.get('duration_seconds')} 秒；掉音比 {technical.get('dropout_ratio')}；削波 {technical.get('clipping_ratio')}",
            f"# 自动歌词检查（仅定位问题，不是审美或音色评分）：",
            f"#   本版：字符相似度 {c['normalized_character_similarity']:.3f}，近似恢复 {c['approximate_lines_recovered']}/{c['lines_total']} 句，替换/漏词/插词 {c['word_edits']['substitution']}/{c['word_edits']['deletion']}/{c['word_edits']['insertion']}",
        ]
        v1_path = OUT / "qa" / f"cover_{slug}_full_v1" / "qa.json"
        if v1_path.is_file():
            v1 = load(v1_path)["comparison"]
            header.append(f"#   full_v1：字符相似度 {v1['normalized_character_similarity']:.3f}，近似恢复 {v1['approximate_lines_recovered']}/{v1['lines_total']} 句")
        if source:
            s = source["comparison"]
            header.append(f"#   同曲原唱干声：字符相似度 {s['normalized_character_similarity']:.3f}，近似恢复 {s['approximate_lines_recovered']}/{s['lines_total']} 句")
        header += [
            "# 注意：SVC 人声本身是 24 kHz（上限 12 kHz），44.1 kHz 混音只保留伴奏的高频。",
            "# 听感与音色一致性仍待人工确认；本版与 full_v1 都是待选 take，不是自动验收。",
        ]
        notes = [w for w in warnings if "no Chinese gloss" not in w]
        if notes:
            header.append("# 歌词对齐提示：" + "；".join(notes))
        stem = f"{slug}_{args.variant}_fixed_voice"
        (FINISHED / f"{stem}.txt").write_text("\n".join(header) + "\n\n" + body, encoding="utf-8")
        (FINISHED / f"{stem}.wav").write_bytes((job / "review_mix.wav").read_bytes())
        (FINISHED / f"{stem}_vocals.wav").write_bytes((job / "converted_vocals.wav").read_bytes())
        made = mix_44k(job, Path(manifest["inputs"]["source"]).stem, FINISHED / f"{stem}_mix_44k.wav")
        print(f"DONE {slug} {args.variant} 44k={made}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
