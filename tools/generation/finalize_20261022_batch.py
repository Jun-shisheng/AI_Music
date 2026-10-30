"""Collect the 2026-10-22 YuE2 batch into one listening delivery.

The batch was stopped by a 17:30 watchdog, so the last run may not have
written index.json. Arc/coverage are taken from index.json when present and
from the batch log otherwise; duration/peak/rate are measured on the delivered
WAV. Nothing is uploaded and the selected/candidates work folders are kept.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "2026-10-22_batch"
FINISHED = OUT / "finished"
BATCHES = [
    {"dir": "epic_en", "voice": "英文御姐", "language": "en",
     "briefs": ROOT / "inputs/_batch_20261022_en",
     "log": ROOT / "logs/yue2_en_20261022_attempt3.log"},
    {"dir": "mandarin_pop", "voice": "中文清甜", "language": "zh",
     "briefs": ROOT / "inputs/_batch_20261022_zh",
     "log": ROOT / "logs/yue2_zh_20261022.out.log"},
    {"dir": "mandarin_pop_2", "voice": "中文清甜", "language": "zh",
     "briefs": ROOT / "inputs/_batch_20261022_zh2",
     "log": ROOT / "logs/yue2_zh2_20261022.out.log"},
]
LOG_LINE = re.compile(
    r"->\s+(?P<slug>.+?)_c1\.wav\s+arc=(?P<arc>.+?)\s+lift=(?P<lift>[\d.]+)\s+"
    r"peak@(?P<peak>[\d.]+)\s+coverage=(?P<coverage>[\d.]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_log(path: Path) -> dict[str, dict]:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOG_LINE.search(line)
        if match:
            values[match.group("slug")] = {
                "verdict": match.group("arc").strip(),
                "lift_db": float(match.group("lift")),
                "peak_at": float(match.group("peak")),
                "coverage": float(match.group("coverage")),
            }
    return values


def load_index(batch_dir: Path) -> dict[str, dict]:
    path = batch_dir / "index.json"
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["title"]: row for row in rows if "title" in row}


def find_lyrics(slug: str) -> Path | None:
    for folder in (ROOT / "inputs/_batch_20261022_en", ROOT / "inputs/_batch_20261022_zh",
                   ROOT / "inputs/_batch_20261022_zh2", ROOT / "inputs"):
        candidate = folder / f"{slug}_lyrics.txt"
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    FINISHED.mkdir(parents=True, exist_ok=True)
    songs = []
    for batch in BATCHES:
        batch_dir = OUT / batch["dir"]
        selected = batch_dir / "selected"
        if not selected.is_dir():
            continue
        index = load_index(batch_dir)
        log_values = parse_log(batch["log"])
        target_dir = FINISHED / batch["voice"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for wav in sorted(selected.glob("*.wav")):
            if wav.stem.endswith("_raw"):
                continue
            slug = wav.stem
            lyrics = find_lyrics(slug)
            info = sf.info(str(wav))
            data, rate = sf.read(str(wav))
            mono = data.mean(axis=1) if data.ndim > 1 else data
            row = index.get(slug, {})
            arc = row.get("arc", {})
            fallback = log_values.get(slug, {})
            verdict = arc.get("verdict") or fallback.get("verdict")
            lift = arc.get("lift_db", fallback.get("lift_db"))
            peak_at = arc.get("peak_at", fallback.get("peak_at"))
            coverage = row.get("diction", fallback.get("coverage"))
            target_wav = target_dir / f"{slug}.wav"
            shutil.copy2(wav, target_wav)
            if lyrics is not None:
                shutil.copy2(lyrics, target_dir / f"{slug}.txt")
            songs.append({
                "slug": slug,
                "language": batch["language"],
                "voice": batch["voice"],
                "engine": "yue2-3b",
                "source_batch": batch["dir"],
                "file": str(target_wav.relative_to(FINISHED)).replace("\\", "/"),
                "lyrics": (f"{slug}.txt" if lyrics is not None else None),
                "duration_seconds": round(info.frames / info.samplerate, 3),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "peak": round(float(np.max(np.abs(mono))), 6),
                "rms": round(float(np.sqrt(np.mean(mono ** 2))), 6),
                "arc_verdict": verdict,
                "lift_db": lift,
                "peak_at": peak_at,
                "lyric_coverage": coverage,
                "weak_lyrics": bool(coverage is not None and coverage < 0.6),
                "sha256": sha256(target_wav),
                "lyrics_sha256": sha256(lyrics) if lyrics is not None else None,
            })
    songs.sort(key=lambda item: (item["voice"], item["slug"]))
    manifest = {
        "schema": 1,
        "date": "2026-10-22",
        "engine": "yue2-3b",
        "line": "native_yue2_48k",
        "delivery_status": "awaiting_listening",
        "uploaded": False,
        "note": "YuE2 native takes; no SVC conversion. The 2026-10-22 A/B showed the 24 kHz SVC path still turns high notes/runs mechanical.",
        "counts": {voice: sum(1 for s in songs if s["voice"] == voice)
                   for voice in sorted({s["voice"] for s in songs})},
        "songs": songs,
    }
    (FINISHED / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(manifest)
    print(json.dumps({"finished": str(FINISHED), "songs": len(songs),
                      "counts": manifest["counts"]}, ensure_ascii=False, indent=2))
    return 0


def _write_readme(manifest: dict) -> None:
    songs = manifest["songs"]
    weak = [s for s in songs if s["weak_lyrics"]]
    lines = [
        "# 2026-10-22 成片交付（YuE2 原生）",
        "",
        f"共 **{len(songs)} 首**，全部为 YuE2 原生完整歌曲（44.1/48 kHz），"
        "没有 SoulX-SVC 换声、没有 EQ/压缩/限幅。",
        "",
        "## 目录",
        "",
        "- `英文御姐/`：英文史诗，成熟女声，强动态",
        "- `中文清甜/`：中文流行，清甜年轻女声",
        "",
        "每首 WAV 旁边有同名 `.txt` 歌词；`manifest.json` 记录时长、峰值、arc、"
        "歌词覆盖和 SHA-256。",
        "",
        "## 试听顺序",
        "",
        "先听中文清甜（新线），再听英文御姐；同目录内按文件名排序。",
        "",
        "## 需要重点听词的成片",
        "",
    ]
    if weak:
        for song in weak:
            lines.append(f"- `{song['file']}`：歌词覆盖 {song['lyric_coverage']:.3f}")
    else:
        lines.append("- 无")
    lines += [
        "",
        "## 自动指标说明",
        "",
        "arc 只描述动态形状（build-up / mild build / early peak），不证明好听；"
        "歌词覆盖来自无提示 ASR，低覆盖可能漏词，也可能是转写误差。",
        "",
        "## 本批缺口",
        "",
        "- 英文 02_unbowed、07_crown_of_thorns 两次 OOM 未出；08–12 未跑。",
        "- 中文 16 之后（17–20）未跑；17:30 watchdog 停机。",
        "- 参考编曲线本轮未重渲染，计划整形/MIDI 对比另行安排。",
        "",
        "## 状态",
        "",
        "`awaiting_listening`。未上传私有仓库；等听审结论后再决定保留/重跑/交付。",
        "",
    ]
    (FINISHED / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
