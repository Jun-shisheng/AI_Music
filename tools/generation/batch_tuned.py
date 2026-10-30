"""Resident-model batch generation with the 12 GB tuned config.

Loads HeartMuLa once and generates every lyrics/tags pair in the input folder,
instead of paying the model load per song. Uses the same settings as
`clean_pipeline.py`: cfg=2.0 (guidance on), codec on GPU, KV cache right-sized to
prompt+frames.

Each song is written as WAV + MP3 + a lyrics TXT, loudness-normalized with a
single linear gain (spectrum-preserving). Existing outputs are skipped, so the
script can be re-run to resume.

Usage:
  python tools/generation/batch_tuned.py --inputs inputs \
      --out-dir outputs/final --seconds 210
  python tools/generation/batch_tuned.py --inputs inputs --out-dir outputs/final \
      --seconds 210 --priority '^1[1-9]_|^20_'   # English titles first
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEART_SRC = PROJECT_ROOT / "vendor" / "heartlib" / "src"
sys.path.insert(0, str(HEART_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for audio_qa

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from audio_qa import tail_degeneracy  # noqa: E402
from heartlib import HeartMuLaGenPipeline  # noqa: E402
import loudness  # noqa: E402

MODEL_ROOT = PROJECT_ROOT / "models" / "heartmula"
MAX_ATTEMPTS = 3  # extra seeds tried when a take ends in a degenerate burst


def order_pairs(pairs: list[tuple], priority: str | None) -> list[tuple]:
    """Move inputs whose slug matches `priority` to the front (stable)."""
    if not priority:
        return pairs
    pat = re.compile(priority)
    return sorted(pairs, key=lambda p: 0 if pat.search(p[0]) else 1)


def normalize(src: Path, dst: Path) -> dict:
    """Spectrum-preserving loudness normalization (see tools/generation/loudness.py).

    The level report is recorded in the manifest: the previous copy-pasted version
    capped the boost at +12 dB and recorded nothing, which left 17 of 20 songs
    below target without anyone able to prove why from the artifacts.
    """
    return loudness.normalize_loudness(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path,
                    default=PROJECT_ROOT / "outputs" / "batch_20260820" / "inputs")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "final")
    ap.add_argument("--seconds", type=int, default=210)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seq-margin", type=int, default=384)
    ap.add_argument("--base-seed", type=int, default=2000)
    ap.add_argument("--priority", default=None,
                    help="regex; matching slugs are generated FIRST (stable order). "
                         "Seed diversity still comes from each slug's sorted position, "
                         "so generation stays reproducible across orderings.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = args.out_dir / "BATCH_STATUS.txt"

    pairs = []
    for lp in sorted(args.inputs.glob("*_lyrics.txt")):
        slug = lp.name[: -len("_lyrics.txt")]
        tp = args.inputs / f"{slug}_tags.txt"
        if tp.exists():
            pairs.append((slug, lp, tp))
    if not pairs:
        print(f"no lyrics/tags pairs found in {args.inputs}")
        return 1

    todo = [(s, i, lp, tp) for i, (s, lp, tp) in enumerate(pairs)
            if not (args.out_dir / s).with_suffix(".wav").exists()]
    todo = order_pairs(todo, args.priority)
    print(f"{len(pairs)} inputs, {len(todo)} to generate into {args.out_dir}")
    if args.priority:
        print("order: " + ", ".join(t[0] for t in todo))
    if not todo:
        status.write_text("all outputs already exist\n", encoding="utf-8")
        return 0

    def note(text: str) -> None:
        status.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)

    note(f"loading model (cfg={args.cfg})")
    pipe = HeartMuLaGenPipeline.from_pretrained(
        str(MODEL_ROOT),
        device={"mula": torch.device("cuda"), "codec": torch.device("cuda")},
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=True,
    )

    manifest = []
    for n, (slug, idx, lp, tp) in enumerate(todo, 1):
        final = args.out_dir / f"{slug}.wav"
        raw = args.out_dir / f".{slug}.raw.wav"
        temperature = [0.90, 0.95, 1.00, 0.88][idx % 4]
        topk = [40, 50, 60, 45][idx % 4]
        result = None
        for attempt in range(MAX_ATTEMPTS):
            seed = args.base_seed + idx * 13 + attempt * 101
            note(f"[{n}/{len(todo)}] {slug} seed={seed} temp={temperature} topk={topk}"
                 + (f"  (retry {attempt})" if attempt else ""))
            t0 = time.time()
            try:
                lyrics = lp.read_text(encoding="utf-8")
                tags = tp.read_text(encoding="utf-8")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                pre = pipe.preprocess({"lyrics": lyrics, "tags": tags}, cfg_scale=args.cfg)
                prompt_len = int(pre["pos"].shape[-1])
                needed = prompt_len + args.seconds * 1000 // 80 + args.seq_margin
                seq_len = max(1024, -(-needed // 512) * 512)
                pipe.mula.backbone.max_seq_len = seq_len
                with torch.no_grad():
                    out = pipe._forward(pre, max_audio_length_ms=args.seconds * 1000,
                                        temperature=temperature, topk=topk,
                                        cfg_scale=args.cfg)
                frames = out["frames"]
                import torchaudio
                wav = pipe.codec.detokenize(frames.to(pipe.codec.device))
                torchaudio.save(str(raw), wav.to(torch.float32).cpu(), 48000)
                pipe._unload()
                level = normalize(raw, final)
                raw.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                err = traceback.format_exc()
                raw.unlink(missing_ok=True)
                final.unlink(missing_ok=True)
                result = {"slug": slug, "error": err.strip().splitlines()[-1]}
                note(f"    FAILED: {err.strip().splitlines()[-1]}")
                continue

            qa = tail_degeneracy(final)
            if qa["degenerate"]:
                note(f"    rejected: degenerate ending ({qa['clicks_per_s']} clicks/s, "
                     f"{qa['tail_share']*100:.0f}% in the tail) -> new seed")
                final.unlink(missing_ok=True)
                result = {"slug": slug, "error": "degenerate ending after retries",
                          "clicks_per_s": qa["clicks_per_s"], "tail_share": qa["tail_share"]}
                continue

            mp3 = final.with_suffix(".mp3")
            y, sr = sf.read(final, always_2d=True)
            sf.write(mp3, y, sr, format="MP3")
            (args.out_dir / f"{slug}.txt").write_text(lyrics, encoding="utf-8")
            dur = sf.info(final).duration
            mins = (time.time() - t0) / 60
            level_note = ""
            if level["boost_capped"]:
                level_note = f"  boost capped at {level['gain_db']:.1f} dB"
            elif level["peak_limited"]:
                level_note = f"  peak-limited ({level['shortfall_db']:.1f} dB short)"
            result = {"slug": slug, "duration_s": round(dur, 2),
                      "frames": int(frames.shape[-1]), "seq_len": seq_len,
                      "seed": seed, "temperature": temperature, "topk": topk,
                      "cfg": args.cfg, "minutes": round(mins, 2),
                      "clicks_per_s": qa["clicks_per_s"], "tail_share": qa["tail_share"],
                      "rms_in": level["rms_in"], "gain_db": level["gain_db"],
                      "rms_out_dbfs": level["rms_out_dbfs"],
                      "boost_capped": level["boost_capped"],
                      "peak_limited": level["peak_limited"]}
            note(f"    ok  {dur:.1f}s  {mins:.1f} min  ({qa['clicks_per_s']} clicks/s)"
                 f"  rms {level['rms_out_dbfs']:.1f} dBFS{level_note}")
            break
        manifest.append(result)

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    done = sum(1 for m in manifest if "duration_s" in m)
    note(f"finished: {done}/{len(todo)} generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
