"""Build a dated delivery folder and push it to the user's private repo.

The user's convention (settled 2026-10-13, extended with the repo step on
2026-10-14): one dated folder holds the songs and their same-stem lyrics, plus
`manifest.json` and the Chinese listening sheet; that folder -- and nothing
else -- is committed to `Jun-shisheng/Music_date`, a private repo the user
listens from. Work files (`candidates/`, `selected/`, tags, tools) are never
pushed.

This wraps the three steps that were done by hand on 2026-10-14 so a finished
batch can be published unattended, which is what the user asked for ("跑完自动
上传"):

  1. `export_delivery.py` -> <repo>/<YYYY-MM-DD_name>/<slug>_<engine>.wav + .txt
     + manifest.json;
  2. `battle_review.py` -> the listening sheet, rewritten for the delivery
     folder (the sheet names batch-dir paths; the delivered files carry the
     engine suffix);
  3. `git add -A && git commit && git push`, skipped when nothing changed, so
     running this twice is harmless.

The clone is kept at `.cache/Music_date_repo` so pushes are incremental and the
history lives outside the project tree.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

FILE_COLUMN = re.compile(r"`([0-9]{2}_[a-z0-9_]+)\.wav`")


def delivery_sheet(text: str, name: str, engine: str, count: int) -> str:
    """Rewrite a batch-dir listening sheet so it describes the delivery folder.

    Two edits: the file column must name the delivered file (`<slug>_<engine>
    .wav`, not the batch dir's `<slug>.wav`), and the header should say which
    delivery it belongs to instead of "battle batch".
    """
    text = FILE_COLUMN.sub(lambda m: f"`{m.group(1)}_{engine}.wav`", text)
    text = text.replace(
        "- 交付文件：`selected/<slug>.wav`（线性响度归一化）；原件：`selected/<slug>_raw.wav`（零处理，保留）",
        f"- 交付文件：`<slug>_{engine}.wav`（线性响度归一化）+ 同名歌词 `.txt`；"
        "批处理目录里另有零处理原件 `selected/<slug>_raw.wav`")
    first, _, rest = text.partition("\n")
    if first.startswith("# "):
        text = f"# 战斗曲试听清单 — {name}（{engine}，{count} 首）" + "\n" + rest
    return text


def commit_message(name: str, count: int, engine: str, slugs: list[str]) -> str:
    return (f"{name}: {count} tracks ({engine})\n\n"
            + "\n".join(f"- {s}" for s in slugs))


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--engine", default="YuE2")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--name", default=None,
                    help="delivery folder name suffix; defaults to the batch dir name")
    ap.add_argument("--repo", type=Path,
                    default=PROJECT_ROOT / ".cache" / "Music_date_repo")
    ap.add_argument("--python", type=Path,
                    default=PROJECT_ROOT / "vendor" / "ACE-Step-1.5" / ".venv" / "Scripts" / "python.exe")
    ap.add_argument("--no-push", action="store_true",
                    help="build the folder and commit locally, but do not push")
    args = ap.parse_args()

    name = args.name or args.batch_dir[0].name
    # A batch dir is usually already dated (outputs/2026-10-14_epic_en), so only
    # prefix the date when it is missing -- otherwise the folder would read
    # 2026-10-14_2026-10-14_epic_en.
    folder_name = name if name.startswith(args.date) else f"{args.date}_{name}"
    out_dir = args.repo / folder_name
    if not (args.repo / ".git").exists():
        print(f"ERROR: no git clone at {args.repo}")
        return 1
    export = run([
        str(args.python), "-I", str(PROJECT_ROOT / "tools" / "generation" / "export_delivery.py"),
        "--engine", args.engine, "--out-dir", str(out_dir),
        "--batch-dir", *[str(b) for b in args.batch_dir],
    ], cwd=PROJECT_ROOT)
    print(export.stdout.strip().splitlines()[-1] if export.stdout.strip() else export.stdout)

    wavs = sorted(p.name for p in out_dir.glob("*.wav"))
    if not wavs:
        print("nothing to publish (no WAVs in the delivery folder)")
        return 1

    sheet_md = args.batch_dir[0] / "review.md"
    review = run([
        str(args.python), "-I", str(PROJECT_ROOT / "tools" / "generation" / "battle_review.py"),
        "--engine", args.engine, "--batch-dir", *[str(b) for b in args.batch_dir],
        "--out", str(sheet_md),
    ], cwd=PROJECT_ROOT)
    print(review.stdout.strip().splitlines()[-1] if review.stdout.strip() else review.stdout)
    sheet_path = out_dir / f"{folder_name}_review.md"
    sheet_path.write_text(
        delivery_sheet(sheet_md.read_text(encoding="utf-8"), name, args.engine, len(wavs)),
        encoding="utf-8")
    print(f"sheet -> {sheet_path.name}", flush=True)

    run(["git", "add", "-A"], cwd=args.repo)
    status = run(["git", "status", "--porcelain"], cwd=args.repo)
    if not status.stdout.strip():
        print("no changes to commit; nothing pushed")
        return 0
    slugs = [Path(w).stem for w in wavs]
    run(["git", "commit", "-q", "-m", commit_message(folder_name, len(wavs), args.engine, slugs)],
        cwd=args.repo)
    if args.no_push:
        print(f"committed locally only ({len(wavs)} tracks); --no-push set")
        return 0
    push = run(["git", "push", "origin", "main"], cwd=args.repo, check=False)
    if push.returncode != 0:
        print(f"PUSH FAILED rc={push.returncode}\n{push.stderr.strip()[-500:]}")
        return 1
    print(f"PUSHED {folder_name}: {len(wavs)} tracks -> {args.repo.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
