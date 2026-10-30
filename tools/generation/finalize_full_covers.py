"""Finalize full-song fixed-voice covers into the dated delivery folder.

For every completed `cover_<slug>_full_v1` job this script:
  * runs the CPU lyric QA on the converted vocals and on the source vocals;
  * writes a bilingual (English line + Chinese line) lyrics sheet with metrics;
  * copies the native 24 kHz mix and dry vocals;
  * builds an optional 44.1 kHz listening mix from the Demucs instrumental
    (the SVC vocal itself stays 24 kHz and is only upsampled).

Nothing here grades timbre or musical quality; the QA text is a proxy for
content preservation.
"""
from __future__ import annotations

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
SLUGS = ["01_hammer_and_hymn", "02_no_white_flag", "03_the_gate_holds",
         "04_iron_rain", "05_drums_below_deck", "06_count_the_living"]

sys.path.insert(0, str(ROOT / "tools/generation"))
import export_delivery as ed  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def qa(audio: Path, lyrics: Path, out_dir: Path) -> dict:
    if (out_dir / "qa.json").is_file():
        return load(out_dir / "qa.json")
    command = [str(PY), str(ROOT / "tools/music_qa.py"), "--audio", str(audio),
               "--lyrics", str(lyrics), "--out-dir", str(out_dir), "--language", "en"]
    code = subprocess.run(command, cwd=ROOT).returncode
    if code != 0:
        raise RuntimeError(f"QA failed for {audio} (exit {code})")
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


def main() -> int:
    FINISHED.mkdir(parents=True, exist_ok=True)
    titles = load(INPUTS / "_titles.json")
    summaries = []
    for slug in SLUGS:
        job_id = f"cover_{slug}_full_v1"
        job = JOBS / job_id
        manifest_path = job / "manifest.json"
        if not manifest_path.is_file():
            print(f"SKIP {job_id}: no manifest")
            continue
        manifest = load(manifest_path)
        if manifest.get("status") not in {"awaiting_listening", "technical_review_required"}:
            print(f"SKIP {job_id}: status {manifest.get('status')}")
            continue
        outputs = manifest.get("outputs", {})
        if "mix" not in outputs or "vocals" not in outputs:
            print(f"SKIP {job_id}: outputs incomplete")
            continue
        lyrics_en = INPUTS / f"{slug}_lyrics.txt"
        lyrics_zh = INPUTS / f"{slug}_lyrics_zh.txt"
        cover_qa = qa(job / "converted_vocals.wav", lyrics_en, OUT / "qa" / job_id)
        source_qa = qa(job / "target_vocals.wav", lyrics_en, OUT / "qa" / f"{job_id}_source")
        c, s = cover_qa["comparison"], source_qa["comparison"]
        technical = manifest.get("technical_qa", {})
        body, warnings = ed.bilingual_body(lyrics_en.read_text(encoding="utf-8"),
                                           lyrics_zh.read_text(encoding="utf-8"))
        title = titles.get(slug, slug)
        header = (
            f"# {title} · {slug} · 固定女声全曲 cover\n"
            f"# 引擎：SoulX-Singer SVC；歌手档案 iron_rain_female_v1\n"
            f"# 音频：{slug}_full_fixed_voice.wav（24 kHz 管线混音）/ _mix_44k.wav（44.1 kHz 试听混音）/ _vocals.wav（干声）\n"
            f"# 时长 {technical.get('duration_seconds')} 秒；掉音比 {technical.get('dropout_ratio')}；削波 {technical.get('clipping_ratio')}\n"
            f"# 自动歌词检查（仅定位问题，不是审美或音色评分）：\n"
            f"#   固定女声版：字符相似度 {c['normalized_character_similarity']:.3f}，近似恢复 {c['approximate_lines_recovered']}/{c['lines_total']} 句，替换/漏词/插词 {c['word_edits']['substitution']}/{c['word_edits']['deletion']}/{c['word_edits']['insertion']}\n"
            f"#   同曲原唱干声：字符相似度 {s['normalized_character_similarity']:.3f}，近似恢复 {s['approximate_lines_recovered']}/{s['lines_total']} 句，替换/漏词/插词 {s['word_edits']['substitution']}/{s['word_edits']['deletion']}/{s['word_edits']['insertion']}\n"
            f"# 注意：SVC 人声本身是 24 kHz（上限 12 kHz），44.1 kHz 混音只保留伴奏的高频。\n"
            f"# 听感与音色一致性仍待人工确认。\n"
        )
        notes = [w for w in warnings if "no Chinese gloss" not in w]
        if notes:
            header += "# 歌词对齐提示：" + "；".join(notes) + "\n"
        txt = FINISHED / f"{slug}_full_fixed_voice.txt"
        txt.write_text(header + "\n" + body, encoding="utf-8")
        mix_native = FINISHED / f"{slug}_full_fixed_voice.wav"
        vocals = FINISHED / f"{slug}_full_fixed_voice_vocals.wav"
        mix_native.write_bytes((job / "review_mix.wav").read_bytes())
        vocals.write_bytes((job / "converted_vocals.wav").read_bytes())
        mix44 = FINISHED / f"{slug}_full_fixed_voice_mix_44k.wav"
        made44 = mix_44k(job, Path(manifest["inputs"]["source"]).stem, mix44)
        summaries.append({"slug": slug, "title": title, "job": job_id,
                          "duration": technical.get("duration_seconds"),
                          "dropout": technical.get("dropout_ratio"),
                          "cover_similarity": c["normalized_character_similarity"],
                          "source_similarity": s["normalized_character_similarity"],
                          "cover_lines": f"{c['approximate_lines_recovered']}/{c['lines_total']}",
                          "source_lines": f"{s['approximate_lines_recovered']}/{s['lines_total']}",
                          "mix_44k": made44})
        print(f"DONE {slug} ({'44k' if made44 else '24k only'})")
    if summaries:
        lines = ["固定女声全曲成品（2026-10-19）", ""]
        for item in summaries:
            lines.append(f"{item['slug']} · {item['title']}")
            lines.append(f"  {item['slug']}_full_fixed_voice.wav / _mix_44k.wav / _vocals.wav / .txt")
            lines.append(f"  时长 {item['duration']} 秒；掉音比 {item['dropout']}；内容代理 固定女声 {item['cover_similarity']:.3f}（{item['cover_lines']}） vs 原唱 {item['source_similarity']:.3f}（{item['source_lines']}）")
            lines.append("")
        lines += [
            "处理：Demucs CPU 分轨 -> SoulX-Singer SVC（固定女声 iron_rain_female_v1，32 steps，CFG 3，seed 42）-> 常数增益电平匹配与直接相加混音。",
            "没有 EQ、降噪、压缩或限幅。",
            "SVC 人声是 24 kHz（上限 12 kHz）；44.1 kHz 混音只把 Demucs 原始伴奏接回并升采样人声，不恢复人声高频。",
            "自动转写只用于定位问题；音色身份、自然度、高音区和分句接缝仍需人工听审。",
        ]
        (FINISHED / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
