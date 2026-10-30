"""Generate full-length versions for the selected review-approved songs."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEART_SRC = PROJECT_ROOT / "vendor" / "heartlib" / "src"
MODEL_ROOT = PROJECT_ROOT / "models" / "heartmula"
INPUT_ROOT = PROJECT_ROOT / "outputs" / "batch_20260820" / "inputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "batch_20260820" / "full_candidates"
STATUS = PROJECT_ROOT / "outputs" / "batch_20260820" / "FULL_BATCH_STATUS.txt"
sys.path.insert(0, str(HEART_SRC))

from heartlib import HeartMuLaGenPipeline  # noqa: E402

import loudness  # noqa: E402

SELECTED = [
    "02_wind_from_the_coast",
    "03_neon_heartbeat",
    "04_unfinished_summer_night",
    "11_glass_hearts_glow",
    "12_run_into_the_blue",
    "13_after_the_rain_comes_gold",
    "14_midnight_rooftop",
    "15_wildfire_in_slow_motion",
    "16_paper_planes_satellites",
    "17_stay_until_sunrise",
    "18_gravity_is_a_lie",
    "19_lavender_city_lights",
    "20_better_than_yesterday",
]


def normalize_loudness(src: Path, dst: Path) -> dict:
    """Loudness-match a take (single implementation: tools/generation/loudness.py)."""
    return loudness.normalize_loudness(src, dst)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(f"loading resident model for {len(SELECTED)} full songs\n", encoding="utf-8")
    pipe = HeartMuLaGenPipeline.from_pretrained(
        str(MODEL_ROOT),
        device={"mula": torch.device("cuda"), "codec": torch.device("cpu")},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=False,
    )
    for position, slug in enumerate(SELECTED, 1):
        final = OUTPUT_ROOT / f"{slug}.wav"
        if final.exists():
            continue
        raw = OUTPUT_ROOT / f".{slug}.raw.wav"
        seed = 1200 + position * 17
        temperature = [0.86, 0.92, 0.98, 0.88][position % 4]
        topk = [40, 50, 55, 45][position % 4]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        STATUS.write_text(f"running {position}/{len(SELECTED)} {slug}\n", encoding="utf-8")
        started = time.time()
        try:
            with torch.no_grad():
                pipe(
                    {"lyrics": str(INPUT_ROOT / f"{slug}_lyrics.txt"), "tags": str(INPUT_ROOT / f"{slug}_tags.txt")},
                    max_audio_length_ms=180_000,
                    save_path=str(raw),
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=1.0,
                )
            normalize_loudness(raw, final)
            raw.unlink(missing_ok=True)
            STATUS.write_text(f"completed {position}/{len(SELECTED)} {slug} {(time.time()-started)/60:.1f} min\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            raw.unlink(missing_ok=True)
            STATUS.write_text(f"failed {position}/{len(SELECTED)} {slug}: {exc}\n", encoding="utf-8")
            return 1
    STATUS.write_text(f"finished {len(SELECTED)} full songs\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
