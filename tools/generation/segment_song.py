"""Segment-based full-song generation for the 8GB GPU.

HeartMuLa cannot generate a full 210s song in one pass on the 8GB card:
measured KV-cache growth reaches 14.36 GB at only 250 frames (16s of audio).
The reliable unit is ~30s (375 frames), which stays under the crash boundary.

Strategy: split the full lyrics into N segments by song-structure boundaries
([主歌]/[Verse] etc.), generate each segment independently with the SAME
seed/temperature/topk for style consistency, then crossfade the segment WAVs.

Caveat: independent segments can drift in key/tempo. This is the pragmatic
path on this hardware; a true continuation (feeding segment k's audio tokens
into segment k+1's prompt) is a future improvement.

Usage:
  python tools/generation/segment_song.py --lyrics x.txt --tags y.txt \
      --title song --seconds 180 --out outputs/song.wav
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / "vendor" / "ACE-Step-1.5" / ".venv" / "Scripts" / "python.exe"
PIPELINE = PROJECT_ROOT / "tools" / "generation" / "clean_pipeline.py"
OUT_DIR = PROJECT_ROOT / "outputs" / "candidates"

# Song-structure boundaries used to split lyrics into segments.
_SECTION_RE = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)


def split_lyrics(lyrics: str, target_segments: int) -> list[str]:
    """Split lyrics into ~target_segments chunks at section boundaries, keeping
    the song order: segments 1..N each get consecutive sections."""
    lines = [ln for ln in lyrics.splitlines()]
    headers = [(i, ln.strip()) for i, ln in enumerate(lines) if _SECTION_RE.match(ln.strip())]
    if len(headers) <= 1:
        n = max(1, target_segments)
        size = max(1, len(lines) // n)
        return ["\n".join(lines[i:i + size]) for i in range(0, len(lines), size)]

    # group headers into target_segments contiguous buckets by index
    n_headers = len(headers)
    bucket_sizes = [n_headers // target_segments + (1 if i < n_headers % target_segments else 0)
                    for i in range(target_segments)]
    buckets: list[list[tuple[int, str]]] = []
    idx = 0
    for size in bucket_sizes:
        buckets.append(headers[idx:idx + size])
        idx += size
    buckets = [b for b in buckets if b]

    segments = []
    for bi, seg in enumerate(buckets):
        start = seg[0][0]
        next_start = buckets[bi + 1][0][0] if bi + 1 < len(buckets) else len(lines)
        segments.append("\n".join(lines[start:next_start]))
    return segments


def generate_segment(lyrics_txt: Path, tags: Path, title: str, seconds: int,
                     idx: int, temperature: float, topk: int, seed: int) -> Path:
    """Generate one segment WAV via clean_pipeline (30s, safe on 8GB).

    The explicit seed is passed through to HeartMuLa so segment reruns are
    reproducible and neighboring segments can use controlled variation.
    """
    cmd = [str(PYTHON), str(PIPELINE),
           "--lyrics", str(lyrics_txt), "--tags", str(tags),
           "--title", f"{title}_seg{idx + 1}", "--candidates", "1",
           "--seconds", str(seconds), "--out-dir", str(OUT_DIR),
           "--base-seed", str(seed), "--base-cfg", "1.0"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    # match only run directories (glob also hits <name>_lyrics.txt's parent path)
    stamp_dirs = sorted(
        (d for d in OUT_DIR.iterdir() if d.is_dir() and d.name.startswith(f"{title}_seg{idx + 1}_")),
        reverse=True,
    )
    if not stamp_dirs:
        raise FileNotFoundError(f"segment {idx + 1} output missing")
    take = stamp_dirs[0] / f"{title}_seg{idx + 1}_c1.wav"
    if not take.exists():
        raise FileNotFoundError(f"segment take missing: {take}")
    return take


def find_existing_segment(title: str, idx: int) -> Path | None:
    """Return the WAV of a previously generated segment, if present."""
    stamp_dirs = sorted(
        (d for d in OUT_DIR.iterdir() if d.is_dir() and d.name.startswith(f"{title}_seg{idx + 1}_")),
        reverse=True,
    )
    for d in stamp_dirs:
        take = d / f"{title}_seg{idx + 1}_c1.wav"
        if take.exists():
            return take
    return None


def crossfade_join(segments: list[Path], output: Path, fade_s: float = 1.0) -> None:
    """Join segment WAVs with true overlapping crossfades (ffmpeg xfade).

    xfade overlaps the tail of segment k with the head of segment k+1 over
    `fade_s` seconds, so there is no silence gap at the seam.
    """
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    inputs = []
    for s in segments:
        inputs += ["-i", str(s)]

    # acrossfade chain (audio): overlap each pair by fade_s.
    filter_complex = ""
    if len(segments) == 1:
        filter_complex = "[0:a]aformat=sample_rates=48000:channel_layouts=stereo[out]"
    else:
        for i in range(len(segments) - 1):
            if i == 0:
                filter_complex += (f"[0:a][1:a]acrossfade=d={fade_s}:c1=tri:c2=tri"
                                   f"[xf{i}]")
            else:
                filter_complex += (f";[xf{i-1}][{i + 1}:a]acrossfade=d={fade_s}:c1=tri:c2=tri"
                                   f"[xf{i}]")
        filter_complex += (f";[xf{len(segments)-2}]aformat="
                           f"sample_rates=48000:channel_layouts=stereo[out]")

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-v", "error", *inputs,
                    "-filter_complex", filter_complex, "-map", "[out]",
                    "-ar", "48000", "-ac", "2", str(output)], check=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Segment-based full-song generation.")
    parser.add_argument("--lyrics", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--seg-seconds", type=int, default=30,
                        help="Seconds per segment (30 is safe on 8GB).")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lyrics = args.lyrics.read_text(encoding="utf-8")
    n_segments = max(1, round(args.seconds / args.seg_seconds))
    parts = split_lyrics(lyrics, n_segments)
    print(f"lyrics split into {len(parts)} segments "
          f"(requested {n_segments} for {args.seconds}s)")

    work = args.lyrics.parent
    seg_wavs = []
    for idx, part in enumerate(parts):
        existing = find_existing_segment(args.title, idx)
        if existing is not None:
            print(f"--- segment {idx + 1}/{len(parts)} (cached) -> {existing.name} ---")
            seg_wavs.append(existing)
            continue
        seg_txt = work / f"{args.title}_seg{idx + 1}_lyrics.txt"
        seg_txt.write_text(part, encoding="utf-8")
        print(f"--- segment {idx + 1}/{len(parts)} ---", flush=True)
        wav = generate_segment(seg_txt, args.tags, args.title,
                               args.seg_seconds, idx, args.temperature, args.topk, args.seed)
        seg_wavs.append(wav)
        print(f"  segment {idx + 1} -> {wav}", flush=True)

    out = args.out or (PROJECT_ROOT / "outputs" / f"{args.title}_full.wav")
    crossfade_join(seg_wavs, out)
    print(f"joined -> {out} ({len(seg_wavs)} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
