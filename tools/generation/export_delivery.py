"""Assemble a dated delivery folder: songs and their lyrics side by side.

Delivery convention (user, 2026-10-13, settled after two corrections):
  * a dated folder `YYYY-MM-DD_<name>` holds the songs **and** their lyrics
    together -- `<slug>_<engine>.wav` next to `<slug>_<engine>.txt`;
  * WAV only, one file per engine version, so a title covered by two engines
    yields two songs and two same-named lyrics files (identical words; the
    header names the engine and the file name pairs them, so the pairing is
    unambiguous even though the text repeats);
  * MP3 is off by default (it duplicates the same audio in a second file); pass
    `--mp3` to also emit libmp3lame -q:a 2 copies;
  * `--split-lyrics` restores the two-folder layout (songs / `_lyrics`) that the
    2026-10-13 battle delivery used.

Sources are the per-batch `selected/` folders produced by battle_batch.py,
acestep_batch_battle.py and yue2_batch_battle.py. Nothing is moved or deleted --
the batch folders stay where the review sheet points at them.

Usage:
  python tools/generation/export_delivery.py --name battle_songs \
      --batch-dir outputs/battle_batch outputs/battle_batch2 outputs/battle_batch_yue2
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The section parser lives with the API service's lyric checks and has no
# third-party imports, so the bilingual pairing reuses it instead of
# duplicating the tag handling.
sys.path.insert(0, str(PROJECT_ROOT / "services" / "api"))

import arc_qa  # noqa: E402
from app.services.lyric_diversity import ordered_sections  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
_SECTION_RE = re.compile(r"\[([^\]]+)\]")


def engine_of(batch: Path) -> str:
    name = batch.name.lower()
    if "yue2" in name:
        return "YuE2"
    if "ace" in name:
        return "ACE-Step"
    return "HeartMuLa"


def brief_folders(root: Path | None = None) -> list[Path]:
    """Every inputs/ subfolder that actually holds briefs.

    Was `inputs/battle_batch*`, which silently skipped the dated folder the
    2026-10-14 epic-English batch uses (`inputs/2026-10-14_epic_en`) -- the
    delivery would then have shipped lyrics-less files named after the slug.
    Matching on content instead of on a name prefix means a new batch folder
    works whatever it is called.
    """
    base = root or (PROJECT_ROOT / "inputs")
    if not base.is_dir():
        # `inputs/` is gitignored, so a fresh clone has none. Callers rely on
        # an empty list here; raising would kill a delivery before it copies
        # its first track.
        return []
    return sorted(p for p in base.iterdir()
                  if p.is_dir() and any(p.glob("*_lyrics.txt")))


def lyrics_index(root: Path | None = None) -> dict[str, Path]:
    """slug -> lyrics file, merged across every brief folder."""
    index: dict[str, Path] = {}
    for folder in brief_folders(root):
        for path in sorted(folder.glob("*_lyrics.txt")):
            index[path.name[: -len("_lyrics.txt")]] = path
    return index


def title_index(root: Path | None = None) -> dict[str, str]:
    """slug -> Chinese display title."""
    index: dict[str, str] = {}
    for folder in brief_folders(root):
        path = folder / "_titles.json"
        if path.exists():
            index.update({k: v for k, v in
                          json.loads(path.read_text(encoding="utf-8")).items()
                          if not k.startswith("_")})
    return index


def gloss_index(root: Path | None = None) -> dict[str, Path]:
    """slug -> `<slug>_lyrics_zh.txt` (the line-by-line Chinese translation)."""
    index: dict[str, Path] = {}
    for folder in brief_folders(root):
        for path in sorted(folder.glob("*_lyrics_zh.txt")):
            index[path.name[: -len("_lyrics_zh.txt")]] = path
    return index


def is_english(lyrics: str) -> bool:
    """True when the sung lines carry no CJK -- i.e. this sheet needs a gloss.

    A Chinese song has nothing to translate, so it keeps the plain layout.
    """
    lines = [ln for ln in lyrics.splitlines()
             if ln.strip() and not ln.strip().startswith("[")]
    return bool(lines) and not any(_CJK.search(ln) for ln in lines)


def parse_sections(text: str) -> list[tuple[str, list[str]]]:
    """Ordered (tag, lines), keeping each tag exactly as the file spells it.

    `lyric_diversity.ordered_sections` lowercases labels, which is right for
    comparing sections but would print `[intro]` into a delivered sheet, so the
    tag text is captured here instead.
    """
    sections: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        tag = _SECTION_RE.fullmatch(stripped)
        if tag:
            sections.append((tag.group(1).strip(), []))
            continue
        if not sections:
            sections.append(("", []))
        sections[-1][1].append(stripped)
    return sections


def bilingual_body(lyrics: str, gloss: str | None) -> tuple[str, list[str]]:
    """English line followed by its Chinese line, section tags kept once.

    The user asked on 2026-10-15 for an English song's lyrics sheet to carry a
    Chinese line under every English line. A wrong pairing would be worse than
    no translation at all, so the alignment is checked per section first; a
    section that does not line up falls back to English block + Chinese block
    and is reported instead of being interleaved blindly.
    """
    en = parse_sections(lyrics)
    if not gloss:
        return lyrics, ["no Chinese gloss available"]
    zh = parse_sections(gloss)

    aligned = ([t.lower() for t, _ in en] == [t.lower() for t, _ in zh]
               and all(len(a) == len(b) for (_, a), (_, b) in zip(en, zh)))
    if not aligned:
        body = (lyrics.rstrip() + "\n\n# ---- 中文（未能逐行对齐，按段另附）----\n\n"
                + gloss.rstrip() + "\n")
        return body, ["gloss is not line-aligned; appended per section"]

    out: list[str] = []
    for (tag, en_lines), (_, zh_lines) in zip(en, zh):
        out.append(f"[{tag}]")
        for en_line, zh_line in zip(en_lines, zh_lines):
            out.append(en_line)
            out.append(zh_line)
        out.append("")
    return "\n".join(out).rstrip() + "\n", []


def lyrics_with_header(slug: str, stem: str, engine: str, row: dict,
                       titles: dict[str, str], body: str,
                       audio_ref: str, bilingual: str | None = None) -> str:
    """Lyrics body preceded by a header naming the engine and the take's numbers.

    Two engines can cover the same title, so a lyrics file on its own must say
    which version it belongs to -- the user asked for the engine to be written
    out explicitly, not just encoded in the file name.
    """
    cn = titles.get(slug, slug)
    arc = row.get("arc") or {}
    header = [
        f"# {cn} · {slug} · 引擎 {engine}",
        f"# 对应音频 {audio_ref}",
        f"# seed {row.get('seed')} | 结构 {arc.get('verdict')} | "
        f"起伏 {arc.get('lift_db')} dB | 高潮位置 {arc.get('peak_at')} | "
        f"桥段回落 {arc.get('dip_before_peak_db')} dB | "
        f"首句 {row.get('first_vocal_s')} s | "
        f"歌词还原 {row.get('diction')}",
    ]
    if bilingual is None:
        header.append("# ---- 歌词 ----")
    else:
        header.append("# ---- 歌词（每行英文下为对应中文）----")
    header.append("")
    return "\n".join(header) + body


def make_mp3(wav: Path, mp3: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                    "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--name", default="battle_songs")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--batch-dir", type=Path, nargs="+", default=[
        PROJECT_ROOT / "outputs" / "battle_batch",
        PROJECT_ROOT / "outputs" / "battle_batch2",
        PROJECT_ROOT / "outputs" / "battle_batch_yue2",
    ])
    ap.add_argument("--engine", default=None,
                    help="engine label for every batch dir, overriding the name "
                         "heuristic -- a batch dir that does not carry the engine "
                         "in its name (e.g. outputs/2026-10-14_epic_en) would "
                         "otherwise be labelled HeartMuLa")
    ap.add_argument("--mp3", action="store_true",
                    help="also write libmp3lame -q:a 2 copies (off by default: "
                         "they duplicate the same audio as a second file)")
    ap.add_argument("--split-lyrics", action="store_true",
                    help="write the lyrics to a sibling <name>_lyrics folder "
                         "instead of next to the songs (the 2026-10-13 layout)")
    ap.add_argument("--lyrics-dir", type=Path, default=None,
                    help="explicit lyrics folder; implies --split-lyrics")
    args = ap.parse_args()

    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / f"{args.date}_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.lyrics_dir or args.split_lyrics:
        lyrics_dir = args.lyrics_dir or out_dir.with_name(out_dir.name + "_lyrics")
        lyrics_dir.mkdir(parents=True, exist_ok=True)
    else:
        lyrics_dir = out_dir            # songs and lyrics in one folder

    lyrics = lyrics_index()
    titles = title_index()
    glosses = gloss_index()
    manifest: list[dict] = []
    for batch in args.batch_dir:
        engine = args.engine or engine_of(batch)
        for wav in sorted((batch / "selected").glob("*.wav")):
            if wav.stem.endswith("_raw"):
                continue                       # zero-processing reference copy
            slug = wav.stem
            stem = f"{slug}_{engine}"
            dst_wav = out_dir / f"{stem}.wav"
            shutil.copy2(wav, dst_wav)

            if args.mp3:
                make_mp3(dst_wav, out_dir / f"{stem}.mp3")

            # The take's own objective numbers, needed both for the lyrics
            # header and for the manifest.
            row: dict = {}
            for result in (batch / "candidates").glob(f"*/result.json"):
                try:
                    candidate = json.loads(result.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if candidate.get("title") == slug:
                    row = candidate

            # Always recompute the arc from the delivered file. The metrics are
            # deterministic functions of the audio, so this is stable, and it
            # means a take whose result.json predates a metric still reports it
            # instead of shipping a header that says "None" (brightness in
            # particular is not stored in result.json at all).
            fresh = arc_qa.analyze(dst_wav)
            row["arc"] = {
                "verdict": fresh["verdict"],
                "lift_db": fresh["lift_db"],
                "peak_at": fresh["peak_at"],
                "quiet_share": fresh["quiet_share"],
                "dip_before_peak_db": fresh["dip_before_peak_db"],
            }
            row["brightness_delta_hz"] = fresh["brightness_delta_hz"]

            # One lyrics file per SONG FILE (strict 1:1, the user's request):
            # both engines sing the same words, so the two copies carry the same
            # lyrics -- only the header identifies which version it belongs to.
            src_lyrics = lyrics.get(slug)
            if src_lyrics:
                same_folder = lyrics_dir == out_dir
                audio_ref = (f"{stem}.wav" if same_folder
                             else f"{out_dir.name}/{stem}.wav")
                text = src_lyrics.read_text(encoding="utf-8")
                # An English song ships with a Chinese line under every English
                # line (user, 2026-10-15). Chinese songs already are Chinese, so
                # they get no gloss and keep the plain layout.
                src_gloss = glosses.get(slug)
                bilingual: str | None = None
                if src_gloss and is_english(text):
                    body, problems = bilingual_body(
                        text, src_gloss.read_text(encoding="utf-8"))
                    bilingual = body
                    for problem in problems:
                        print(f"  {stem}: bilingual pairing -- {problem}",
                              flush=True)
                (lyrics_dir / f"{stem}.txt").write_text(
                    lyrics_with_header(slug, stem, engine, row, titles,
                                       bilingual if bilingual else text,
                                       audio_ref, bilingual=bilingual),
                    encoding="utf-8")

            manifest.append({
                "file": dst_wav.name,
                "engine": engine,
                "title": slug,
                "seed": row.get("seed"),
                "arc": (row.get("arc") or {}).get("verdict"),
                "lift_db": (row.get("arc") or {}).get("lift_db"),
                "peak_at": (row.get("arc") or {}).get("peak_at"),
                "quiet_share": (row.get("arc") or {}).get("quiet_share"),
                "dip_before_peak_db": (row.get("arc") or {}).get("dip_before_peak_db"),
                "diction_coverage": row.get("diction"),
                "brightness_delta_hz": row.get("brightness_delta_hz"),
                "climax_brighter": (row.get("brightness_delta_hz") or 0) > 150,
                "lyrics_txt": (f"{lyrics_dir.name}/{stem}.txt"
                               if src_lyrics else None),
            })
            print(f"  {stem}", flush=True)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(manifest)} tracks -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
