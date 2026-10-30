"""Build one upload-ready folder from the finished batches, split by sung language.

The batch folders are organised by the line they were requested for, not by what
the model actually sang: the 2026-10-22/23 "Mandarin" briefs include ten English
songs. This packages by the text in the delivered lyrics file instead, so the
private library never files an English song under 中文清甜.

English songs must already carry the bilingual sheet (apply_bilingual_lyrics.py
interleaves it); this script only checks and reports, it does not translate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_delivery as ed  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(source: Path) -> list[dict]:
    """Rows from a finished/ folder, with absolute wav/txt paths."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for song in manifest["songs"]:
        wav = source / song["file"]
        txt = (source / song["voice"] / song["lyrics"]) if song.get("lyrics") else None
        rows.append({"song": song, "wav": wav, "txt": txt})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path, action="append", required=True,
                    help="a finished/ folder with manifest.json (repeatable)")
    ap.add_argument("--retake-source", type=Path, action="append", default=[],
                    help="finished/ folder whose songs go to 重跑候选/ with a "
                         "_retake suffix")
    ap.add_argument("--date", default="2026-10-23")
    args = ap.parse_args()

    out: Path = args.out
    groups = {
        "英文御姐": out / "英文御姐",
        "中文清甜": out / "中文清甜",
        "重跑候选": out / "重跑候选",
    }
    for folder in groups.values():
        folder.mkdir(parents=True, exist_ok=True)

    # A bilingual sheet contains Chinese lines, so is_english() on the delivered
    # TXT would call an English song Chinese. The gloss sidecar is the reliable
    # signal: one exists for every English title and for none of the Chinese ones.
    glosses = ed.gloss_index()

    songs: list[dict] = []
    problems: list[str] = []

    def add(source: Path, retake: bool) -> None:
        for item in collect(source):
            song = item["song"]
            txt = item["txt"]
            text = txt.read_text(encoding="utf-8") if txt else ""
            english = song["slug"] in glosses or ed.is_english(text)
            if english and not any("一" <= ch <= "鿿" for ch in text):
                problems.append(f"{song['slug']}: English but no Chinese gloss")
            folder = "重跑候选" if retake else ("英文御姐" if english else "中文清甜")
            stem = song["slug"] + ("_retake" if retake else "")
            dst_wav = groups[folder] / f"{stem}.wav"
            shutil.copy2(item["wav"], dst_wav)
            dst_txt = groups[folder] / f"{stem}.txt"
            if txt:
                shutil.copy2(txt, dst_txt)
            row = dict(song)
            row.update({
                "file": f"{folder}/{stem}.wav",
                "lyrics": f"{stem}.txt" if txt else None,
                "language": "en" if english else "zh",
                "voice": folder,
                "sha256": sha256(dst_wav),
                "lyrics_sha256": sha256(dst_txt) if txt else None,
            })
            songs.append(row)

    for source in args.source:
        add(source, retake=False)
    for source in args.retake_source:
        add(source, retake=True)

    songs.sort(key=lambda s: (s["voice"], s["slug"]))
    manifest = {
        "schema": 1,
        "date": args.date,
        "engine": "yue2-3b",
        "line": "native_yue2_48k",
        "delivery_status": "awaiting_listening",
        "note": ("Native YuE2 takes, no SVC. Folders follow the sung language, "
                 "not the batch that requested it. English sheets carry a "
                 "Chinese line under every English line."),
        "counts": {voice: sum(1 for s in songs if s["voice"] == voice)
                   for voice in sorted({s["voice"] for s in songs})},
        "songs": songs,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [f"# {args.date} 交付（YuE2 原生）", "",
             f"共 **{len(songs)} 首**。目录按**实际演唱语言**划分，不是按批次。", "",
             f"- `英文御姐/`：{manifest['counts'].get('英文御姐', 0)} 首，"
             "歌词 TXT 每行英文下面一行中文",
             f"- `中文清甜/`：{manifest['counts'].get('中文清甜', 0)} 首",
             f"- `重跑候选/`：{manifest['counts'].get('重跑候选', 0)} 首，"
             "自动化指标不达标曲目的新 take，供对比后再决定保留哪版", ""]
    if problems:
        lines.append("## 待处理")
        lines += [f"- {p}" for p in problems]
        lines.append("")
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"out": str(out), "counts": manifest["counts"],
                      "problems": problems}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
