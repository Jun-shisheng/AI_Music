"""Explicit pinned downloads + content verification. Does not load CUDA or install packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from .assets import ASSETS
except ImportError:
    from assets import ASSETS


def verify(path:Path,expected:dict)->dict:
    if path.stat().st_size!=expected["size"]:
        raise ValueError(f"Wrong size for {path}: {path.stat().st_size} != {expected['size']}")
    sha=hashlib.sha256()
    git=hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            sha.update(block)
            git.update(block)
    if expected.get("sha256") and sha.hexdigest()!=expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    if expected.get("git_blob_sha1") and git.hexdigest()!=expected["git_blob_sha1"]:
        raise ValueError(f"Git blob digest mismatch: {path}")
    return dict(path=str(path.resolve()),size=path.stat().st_size,sha256=sha.hexdigest(),verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,default=Path(__file__).resolve().parents[2]/"models/voice_studio")
    parser.add_argument("--include-svs",action="store_true",help="Also fetch the separate 2.82GB SVS checkpoint")
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    manifest=dict(status="verifying" if args.verify_only else "downloading",assets={})
    args.models.mkdir(parents=True,exist_ok=True)
    try:
        if not args.verify_only:
            from huggingface_hub import hf_hub_download
        for repo,spec in ASSETS.items():
            record=manifest["assets"][repo]=dict(revision=spec["revision"],files={})
            for name,expected in spec["files"].items():
                if name=="model.pt" and not args.include_svs:
                    continue
                path=args.models/spec["directory"]/name
                if not args.verify_only:
                    print(f"Downloading {repo}@{spec['revision']}: {name}",flush=True)
                    path=Path(hf_hub_download(repo,name,revision=spec["revision"],local_dir=str(path.parent)))
                record["files"][name]=verify(path,expected)
                print(f"Verified {path}",flush=True)
        manifest["status"]="verified"
    except Exception as exc:
        manifest.update(status="failed",error=str(exc))
        raise
    finally:
        (args.models/"verified_assets.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
