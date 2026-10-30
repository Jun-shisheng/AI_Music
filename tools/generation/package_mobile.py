"""Build a flat phone-ready folder from a YuE2 batch.

  NN_<Chinese title>.mp3   ~190 kbps, tagged with title/artist/album
  NN_<Chinese title>.txt   the lyric sheet, section tags intact
  <审核文档>.md             written by hand next to it

The numbered prefix keeps the album order in every player; the Chinese name
is what the phone shows.

Usage:
  python tools/generation/package_mobile.py \
      --batch outputs/2026-10-29_film_ref2 \
      --inputs inputs/2026-10-29_film_ref2 \
      --dest outputs/2026-10-29_交付_2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"


def title_map(inputs: Path) -> dict[str, str]:
    path = inputs / "_titles.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--album", default="2026-10-29")
    args = ap.parse_args()

    titles = title_map(args.inputs)
    args.dest.mkdir(parents=True, exist_ok=True)
    songs = sorted((args.batch / "selected").glob("*.wav"))
    songs = [p for p in songs if not p.stem.endswith("_raw")]
    if not songs:
        print("no selected wavs found")
        return 1

    for wav in songs:
        slug = wav.stem
        num = slug.split("_")[0]
        title = titles.get(slug, slug)
        mp3 = args.dest / f"{num}_{title}.mp3"
        listen = args.batch / "review" / "listen" / f"{slug}.mp3"
        if listen.exists():
            shutil.copy2(listen, mp3)
        else:
            subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error",
                            "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                            "-q:a", "2", str(mp3)], check=True)
        subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(mp3), "-c", "copy",
                        "-metadata", f"title={title}",
                        "-metadata", "artist=YuE2",
                        "-metadata", f"album={args.album}",
                        "-metadata", f"track={num}",
                        str(mp3.with_suffix(".tmp.mp3"))], check=True)
        mp3.with_suffix(".tmp.mp3").replace(mp3)

        lyr = args.inputs / f"{slug}_lyrics.txt"
        if lyr.exists():
            shutil.copy2(lyr, args.dest / f"{num}_{title}.txt")
        print(f"{num}_{title}: mp3+txt")

    total = sum(p.stat().st_size for p in args.dest.iterdir() if p.is_file())
    print(f"{len(songs)} songs, {total / 1e6:.1f} MB -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
