"""Generate the 20-song sample batch with one resident HeartMuLa model."""
from __future__ import annotations

import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEART_SRC = PROJECT_ROOT / "vendor" / "heartlib" / "src"
MODEL_ROOT = PROJECT_ROOT / "models" / "heartmula"
INPUT_ROOT = PROJECT_ROOT / "outputs" / "batch_20260820" / "inputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "batch_20260820" / "candidates"
STATUS = PROJECT_ROOT / "outputs" / "batch_20260820" / "BATCH_STATUS.txt"
import sys
sys.path.insert(0, str(HEART_SRC))

from heartlib import HeartMuLaGenPipeline  # noqa: E402

import loudness  # noqa: E402


def normalize_loudness(src: Path, dst: Path) -> dict:
    """Loudness-match a take (single implementation: tools/generation/loudness.py)."""
    return loudness.normalize_loudness(src, dst)


def read_index() -> list[tuple[str, str]]:
    rows = []
    for line in (INPUT_ROOT / "INDEX.txt").read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            slug, title, _style = line.split("\t", 2)
            rows.append((slug, title))
    return rows


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_index()
    STATUS.write_text(f"resident model loading for {len(rows)} songs\n", encoding="utf-8")
    pipe = HeartMuLaGenPipeline.from_pretrained(
        str(MODEL_ROOT),
        device={"mula": torch.device("cuda"), "codec": torch.device("cpu")},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=False,
    )
    for index, (slug, title) in enumerate(rows):
        final = OUTPUT_ROOT / f"{slug}.wav"
        if final.exists():
            continue
        raw = OUTPUT_ROOT / f".{slug}.raw.wav"
        seed = 520 + index * 11
        temperature = [0.82, 0.94, 1.06, 0.88][index % 4]
        topk = [35, 50, 65, 45][index % 4]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        started = time.time()
        STATUS.write_text(f"running {index + 1}/{len(rows)} {slug}\n", encoding="utf-8")
        try:
            with torch.no_grad():
                pipe(
                    {"lyrics": str(INPUT_ROOT / f"{slug}_lyrics.txt"), "tags": str(INPUT_ROOT / f"{slug}_tags.txt")},
                    max_audio_length_ms=30_000,
                    save_path=str(raw),
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=1.0,
                )
            normalize_loudness(raw, final)
            raw.unlink(missing_ok=True)
            STATUS.write_text(
                f"completed {index + 1}/{len(rows)} {slug} {title} {(time.time()-started)/60:.1f} min\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            raw.unlink(missing_ok=True)
            STATUS.write_text(f"failed {index + 1}/{len(rows)} {slug}: {exc}\n", encoding="utf-8")
    STATUS.write_text(f"finished {len(rows)} songs\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
