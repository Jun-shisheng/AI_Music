"""Collect the 2026-10-23 gap-fill batch into one listening delivery.

This batch runs one OS process per title (see start_yue2_batch_20261023.ps1), so
there is no single run log; arc/coverage come from each out-dir's cumulative
index.json, with a scan of the per-title logs as fallback when a row is missing.
Nothing is uploaded and the selected/candidates work folders are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "2026-10-23_batch"
FINISHED = OUT / "finished"
BATCHES = [
    {"dir": "epic_en", "voice": "英文御姐", "language": "en",
     "briefs": ROOT / "inputs/_batch_20261022_en",
     "log_glob": "yue2_20261023_epic_en_*.out.log"},
    {"dir": "mandarin_pop", "voice": "中文清甜", "language": "zh",
     "briefs": ROOT / "inputs/_batch_20261022_zh2",
     "log_glob": "yue2_20261023_mandarin_pop_*.out.log"},
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


def parse_logs(pattern: str) -> dict[str, dict]:
    values: dict[str, dict] = {}
    for path in sorted((ROOT / "logs").glob(pattern)):
        if not path.is_file():
            continue
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


def find_lyrics(batch: dict, slug: str) -> Path | None:
    for folder in (batch["briefs"], ROOT / "inputs"):
        candidate = folder / f"{slug}_lyrics.txt"
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help="batch root holding epic_en/ and mandarin_pop/ "
                         "(default: outputs/2026-10-23_batch)")
    ap.add_argument("--log-prefix", default="yue2_20261023",
                    help="per-title log prefix in logs/ (default: the run above; "
                         "use yue2_retake_20261023 for the retake tree)")
    ap.add_argument("--label", default="补缺批次",
                    help="README title label")
    args = ap.parse_args()
    out: Path = args.out_dir
    finished = out / "finished"
    finished.mkdir(parents=True, exist_ok=True)
    songs = []
    for batch in BATCHES:
        batch = dict(batch)
        batch["log_glob"] = (f"{args.log_prefix}_{batch['dir']}_*.out.log")
        batch_dir = out / batch["dir"]
        selected = batch_dir / "selected"
        if not selected.is_dir():
            continue
        index = load_index(batch_dir)
        log_values = parse_logs(batch["log_glob"])
        target_dir = finished / batch["voice"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for wav in sorted(selected.glob("*.wav")):
            if wav.stem.endswith("_raw"):
                continue
            slug = wav.stem
            lyrics = find_lyrics(batch, slug)
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
                "file": str(target_wav.relative_to(finished)).replace("\\", "/"),
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
        "date": "2026-10-23",
        "engine": "yue2-3b",
        "line": "native_yue2_48k",
        "delivery_status": "awaiting_listening",
        "uploaded": False,
        "note": ("Gap-fill batch: the English and Mandarin titles the 2026-10-22 "
                 "run missed. Native YuE2, no SVC conversion. One process per "
                 "title with --gpu-mem-fraction 0.90 (see progress.md)."),
        "counts": {voice: sum(1 for s in songs if s["voice"] == voice)
                   for voice in sorted({s["voice"] for s in songs})},
        "songs": songs,
    }
    (finished / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_readme(manifest, finished, args.label)
    print(json.dumps({"finished": str(finished), "songs": len(songs),
                      "counts": manifest["counts"]}, ensure_ascii=False, indent=2))
    return 0


def _write_readme(manifest: dict, finished: Path, label: str = "补缺批次") -> None:
    songs = manifest["songs"]
    weak = [s for s in songs if s["weak_lyrics"]]
    lines = [
        f"# 2026-10-23 {label}（YuE2 原生）",
        "",
        f"共 **{len(songs)} 首**。"
        "全部为 YuE2 原生完整歌曲，没有 SVC 换声、没有 EQ/压缩/限幅。",
        "",
        "## 目录",
        "",
        "- `英文御姐/`：英文史诗，成熟女声，强动态",
        "- `中文清甜/`：中文流行，清甜年轻女声",
        "",
        "每首 WAV 旁边有同名 `.txt` 歌词；`manifest.json` 记录时长、峰值、arc、"
        "歌词覆盖和 SHA-256。",
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
        "arc 只描述动态形状，不证明好听；歌词覆盖来自无提示 ASR，低覆盖可能漏词，"
        "也可能是转写误差。",
        "",
        "## 本轮参数变化",
        "",
        "- 每首一个独立进程（干净分配器）。",
        "- `--gpu-mem-fraction 0.90`：把进程内存上限从 YuE2 默认的 10 GiB 抬到 "
        "10.8 GiB，这是 10-22 长 plan OOM 的真正原因。",
        "- plan 与 render 之间、每次失败重试前执行 `torch.cuda.empty_cache()`。",
        "",
        "## 状态",
        "",
        "`awaiting_listening`。未上传私有仓库；等听审结论后再决定保留/重跑/交付。",
        "",
    ]
    (finished / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
