"""Silent background download of the YuE2 stack from the HuggingFace mirror.

Uses curl rather than huggingface_hub on purpose: hf_hub 0.36 in the ACE venv
fails its HEAD probe against hf-mirror.com (the Xet/curl transport path is not
usable here) while plain HTTP against the mirror works fine. curl also gives us
resume (-C -) and retries for free.

Each repo is written to its own directory under --dest so the downloaded
third-party weights stay isolated from project code.

Usage (always launched detached, never in the foreground):
    python download_models_yue2.py --dest D:\\Model\\AI_Music\\models\\yue2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# (repo_id, destination subdirectory). LoRAs are KB-to-MB, the base model is 7.3 GB.
REPOS: list[tuple[str, str]] = [
    ("m-a-p/YuE2-3B", "YuE2-3B"),
    ("m-a-p/YuE2-Vae", "YuE2-Vae"),
    ("monsterovich/yue2-steps-from-hell", "loras/tsfh-steps-from-hell"),
    ("Mothersuperior/YuE2-instrumental-cot-full-loras", "loras/instrumental-cot"),
]

# Demo renders and licence blurbs: not weights, not needed for inference.
SKIP_SUFFIXES = (".mp3", ".flac", ".wav")


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def list_files(repo: str) -> list[tuple[str, int]]:
    """Return (path, size) for every file in the repo, largest last."""
    url = f"{ENDPOINT}/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = [(s["rfilename"], int(s.get("size") or 0)) for s in data.get("siblings", [])]
    return files


def fetch(repo: str, name: str, dest: Path, size: int) -> bool:
    target = dest / name
    if target.exists() and size and target.stat().st_size == size:
        print(f"[{stamp()}]   cached {name}", flush=True)
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ENDPOINT}/{repo}/resolve/main/{name}"
    cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
           "-C", "-", "--silent", "--show-error", "-o", str(target), url]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{stamp()}]   FAIL {name}: {proc.stderr.strip()[:200]}", flush=True)
        return False
    got = target.stat().st_size
    dt = max(time.time() - t0, 0.01)
    print(f"[{stamp()}]   ok   {name} {got / 1e6:.1f} MB "
          f"({got / 1e6 / dt:.1f} MB/s)", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--only", default=None,
                    help="only download repos whose id contains this substring")
    args = ap.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{stamp()}] endpoint={ENDPOINT}", flush=True)
    print(f"[{stamp()}] dest={dest}", flush=True)

    failures: list[str] = []
    for repo, sub in REPOS:
        if args.only and args.only not in repo:
            continue
        print(f"[{stamp()}] START {repo} -> {dest / sub}", flush=True)
        t0 = time.time()
        try:
            files = list_files(repo)
        except Exception as exc:
            print(f"[{stamp()}] FAIL {repo}: cannot list files: {exc!r}", flush=True)
            failures.append(repo)
            continue
        # Small files first so an interrupted run still leaves a usable config;
        # the multi-GB weights come last.
        todo = [(n, s) for n, s in sorted(files, key=lambda x: x[1])
                if not n.endswith(SKIP_SUFFIXES)]
        print(f"[{stamp()}]   {len(todo)} files, "
              f"{sum(s for _, s in todo) / 1e9:.2f} GB", flush=True)
        bad = [n for n, s in todo if not fetch(repo, n, dest / sub, s)]
        if bad:
            failures.append(repo)
            print(f"[{stamp()}] FAIL {repo}: {len(bad)} files failed", flush=True)
        else:
            total = sum(f.stat().st_size for f in (dest / sub).rglob("*") if f.is_file())
            print(f"[{stamp()}] DONE  {repo} {total / 1e9:.2f} GB "
                  f"in {time.time() - t0:.0f}s", flush=True)

    print(f"[{stamp()}] ALL DONE failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
