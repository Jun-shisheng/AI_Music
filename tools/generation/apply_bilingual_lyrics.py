"""Rewrite delivered English lyrics sheets so each Chinese line sits under its English line.

The finished/ folders carry the sung text as `<slug>.txt`. The line-by-line
Chinese gloss lives next to the brief as `<slug>_lyrics_zh.txt` (written by
`epic_lyrics_gen.py --gloss-only`). This pairs them with the same alignment
check export_delivery uses, without touching audio or the folder layout.

Chinese songs are left alone: there is nothing to translate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_delivery as ed  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDERS = [
    PROJECT_ROOT / "outputs/2026-10-22_batch/finished",
    PROJECT_ROOT / "outputs/2026-10-23_batch/finished",
    PROJECT_ROOT / "outputs/2026-10-23_review_20261022/01_成片_21首",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*", type=Path, default=None,
                    help="delivery folders to rewrite (default: the three "
                         "2026-10-22/23 finished trees)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    glosses = ed.gloss_index()
    folders = args.folders or DEFAULT_FOLDERS
    rewritten = untouched = missing = 0
    for folder in folders:
        if not folder.is_dir():
            print(f"skip (missing): {folder}")
            continue
        for txt in sorted(folder.rglob("*.txt")):
            slug = txt.stem
            text = txt.read_text(encoding="utf-8")
            if not ed.is_english(text):
                untouched += 1
                continue
            gloss = glosses.get(slug)
            if gloss is None:
                print(f"  {slug}: no Chinese gloss -- left as English")
                missing += 1
                continue
            body, problems = ed.bilingual_body(
                text, gloss.read_text(encoding="utf-8"))
            for problem in problems:
                print(f"  {slug}: {problem}")
            if args.dry_run:
                print(f"  {slug}: would rewrite ({len(body)} chars)")
                continue
            txt.write_text(body, encoding="utf-8")
            rewritten += 1
    print(f"rewritten {rewritten}, already-Chinese {untouched}, no-gloss {missing}"
          + (" (dry run)" if args.dry_run else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
