"""Copy a finished fixed-voice delivery into the private repo and push it.

Deliverables only: WAVs, same-stem TXT sheets, manifest.json and the Chinese
review sheet. The manifest's lyrics paths are rewritten to the repo folder name
so the folder reads correctly from a clone. Pushing is skipped when the commit
would be empty, which makes the step idempotent.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], cwd: Path | None = None, check: bool = True):
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--folder-name", required=True)
    ap.add_argument("--repo", type=Path,
                    default=PROJECT_ROOT / ".cache" / "Music_date_repo")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    if not (args.repo / ".git").exists():
        print(f"ERROR: no git clone at {args.repo}")
        return 1
    out = args.repo / args.folder_name
    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in sorted(args.source_dir.iterdir()):
        if item.is_file() and item.suffix.lower() in {".wav", ".txt", ".md", ".json"}:
            shutil.copy2(item, out / item.name)
            copied += 1
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            name = Path(entry.get("lyrics_txt", "")).name
            if name:
                entry["lyrics_txt"] = f"{args.folder_name}/{name}"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wavs = sorted(p.name for p in out.glob("*.wav"))
    if not wavs:
        print("nothing to publish (no WAVs)")
        return 1
    print(f"copied {copied} files; {len(wavs)} WAVs in {out.name}", flush=True)
    run(["git", "add", "-A"], cwd=args.repo)
    status = run(["git", "status", "--porcelain"], cwd=args.repo)
    if not status.stdout.strip():
        print("no changes to commit")
        return 0
    message = (f"{args.folder_name}: {len(wavs)} fixed-voice tracks\n\n"
               + "\n".join(f"- {name}" for name in wavs))
    run(["git", "commit", "-q", "-m", message], cwd=args.repo)
    if args.no_push:
        print("committed locally only (--no-push)")
        return 0
    push = run(["git", "push", "origin", "main"], cwd=args.repo, check=False)
    if push.returncode != 0:
        print(f"PUSH FAILED rc={push.returncode}\n{push.stderr.strip()[-500:]}")
        return 1
    print(f"PUSHED {args.folder_name}: {len(wavs)} WAVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
