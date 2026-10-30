"""Frame-level repair for AI vocal artifacts on a separated vocal stem.

Targets only the defect regions detected by artifact_segments.detect_events
instead of filtering the whole track (whole-track cleanup has been shown to
regress quality). Defect frames are characterised by an abnormal 4-10 kHz
broadband energy burst on top of an otherwise normal voice fundamental.

Repairs:

- harsh_machine_noise frames: STFT-domain soft attenuation of the 4-10 kHz
  band inside the defect region only. Below 3.5 kHz (fundamental and harmonics)
  is untouched, so timbre and intelligibility are preserved.
- abrupt_level_drop frames: gain recovery to the surrounding level, aligned to
  the detected 50 ms frame with short ramps.
- possible_dropout frames: gain recovery when residual signal exists; silent
  gaps are left untouched (structural defects must be rejected, not patched).

All detected events are repaired (no top-N truncation) so full-length songs
get the same coverage as short candidates.

Usage:
    python frame_repair.py <vocals.wav> <output.wav> [--report report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))

from artifact_segments import detect_events  # noqa: E402

MERGE_GAP_S = 0.15
NOISE_BAND_LOW = 4000.0
NOISE_BAND_HIGH = 10000.0
NOISE_ATTEN = 0.22
BAND_EDGE_HZ = 1000.0
RAMP_S = 0.004
CONTEXT_S = 0.30
MAX_GAIN_DB = 10.0
LEVEL_DROP_MIN_DB = 4.0
DETECT_FRAME_S = 0.05
N_FFT = 2048
HOP = 512


def _events_to_regions(events: list[dict], duration_s: float) -> list[dict]:
    """Merge nearby events into repair regions."""
    points = []
    for e in events:
        t = float(e["time_seconds"])
        points.append((max(0.0, t), min(duration_s, t + DETECT_FRAME_S), e["type"]))
    points.sort()
    regions = []
    for start, end, etype in points:
        if regions and start <= regions[-1]["end"] + MERGE_GAP_S:
            regions[-1]["end"] = max(regions[-1]["end"], end)
            regions[-1]["types"].add(etype)
        else:
            regions.append({"start": start, "end": end, "types": {etype}})
    return regions


def _rms(seg: np.ndarray) -> float:
    if len(seg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg**2)))


def _noise_mask(sr: int) -> np.ndarray:
    import librosa

    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    mask = np.ones(len(freqs))
    band = (freqs >= NOISE_BAND_LOW) & (freqs <= NOISE_BAND_HIGH)
    mask[band] = NOISE_ATTEN
    low_edge = (freqs >= NOISE_BAND_LOW - BAND_EDGE_HZ) & (freqs < NOISE_BAND_LOW)
    high_edge = (freqs > NOISE_BAND_HIGH) & (freqs <= NOISE_BAND_HIGH + BAND_EDGE_HZ)
    mask[low_edge] = np.linspace(1.0, NOISE_ATTEN, low_edge.sum())
    mask[high_edge] = np.linspace(NOISE_ATTEN, 1.0, high_edge.sum())
    return mask


def _gain_recover(y: np.ndarray, sr: int, start: float, end: float) -> tuple[np.ndarray, dict]:
    lo = max(0, int(start * sr))
    hi = min(len(y), int(end * sr))
    seg = y[lo:hi]
    current = _rms(seg)
    if current < 1e-6:
        return y, {"status": "silent_skipped"}
    ctx_lo = max(0, int((start - CONTEXT_S) * sr))
    ctx_hi = min(len(y), int((end + CONTEXT_S) * sr))
    target = max(_rms(y[ctx_lo:lo]), _rms(y[hi:ctx_hi]))
    if target < 1e-6:
        return y, {"status": "no_context"}
    gain_db = min(MAX_GAIN_DB, 20 * np.log10(target / current))
    if gain_db < LEVEL_DROP_MIN_DB:
        return y, {"status": "not_a_drop", "gain_db": round(gain_db, 2)}
    gain = 10 ** (gain_db / 20)
    out = y.copy()
    ramp_len = max(1, int(RAMP_S * sr))
    ramp = np.ones(seg.shape[0])
    if len(ramp) > 2 * ramp_len:
        ramp[:ramp_len] = np.linspace(1.0, gain, ramp_len)
        ramp[ramp_len : len(ramp) - ramp_len] = gain
        ramp[len(ramp) - ramp_len :] = np.linspace(gain, 1.0, ramp_len)
    else:
        ramp[:] = gain
    out[lo:hi] = seg * ramp
    return out, {"status": "applied", "gain_db": round(gain_db, 2)}


def _isolated_harsh_regions(input_wav: Path, duration_s: float,
                            zcr_thresh: float = 0.38) -> list[dict]:
    """Detect ISOLATED harsh-burst frames only.

    The zcr harsh detector cannot separate sibilants from machine noise: real
    human vocal references trigger 14-149 'harsh' events per file because
    s/sh/ch naturally have high zcr. Measured separation: AI harsh frames are
    68% isolated single-frame bursts, human sibilant runs only 27% isolated
    (median run length 2). So a frame whose NEIGHBORS are also high-zcr is a
    sibilant (diction) and must be preserved; a frame that spikes alone is a
    machine-noise burst worth repairing.
    """
    import array
    import math
    import subprocess

    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(input_wav), "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        stderr=subprocess.DEVNULL,
    )
    samples = array.array("h")
    samples.frombytes(raw)
    fs = 800
    rms_values, zcr_values = [], []
    for start in range(0, len(samples), fs):
        frame = samples[start:start + fs]
        if not frame:
            continue
        rms_values.append(math.sqrt(sum((v / 32768.0) ** 2 for v in frame) / len(frame)))
        crossings = 0
        prev = frame[0]
        for cur in frame[1:]:
            if (prev < 0 <= cur) or (prev >= 0 > cur):
                crossings += 1
            prev = cur
        zcr_values.append(crossings / max(1, len(frame) - 1))

    floor = max(0.012, sorted(rms_values)[round(len(rms_values) * 0.35)] * 0.75)
    flags = [(r > floor and z > zcr_thresh) for r, z in zip(rms_values, zcr_values)]
    frame_seconds = fs / 16000.0
    regions = []
    for i, flag in enumerate(flags):
        prev_flag = flags[i - 1] if i > 0 else False
        next_flag = flags[i + 1] if i + 1 < len(flags) else False
        if flag and not prev_flag and not next_flag:
            start = i * frame_seconds
            regions.append({
                "start": max(0.0, start),
                "end": min(duration_s, start + DETECT_FRAME_S),
                "types": {"harsh_machine_noise"},
            })
    return regions


def repair_vocals(input_wav: Path, output_wav: Path, repair_rounds: int = 1,
                  isolation_only: bool = False) -> dict:
    """Repair vocal artifacts on a separated vocal stem.

    repair_rounds=1 is the conservative single pass. Two rounds are measurably
    better for AI vocal harshness: one pass lowers zcr ~0.65 -> 0.56-0.68 (still
    above the audible 0.55 threshold and above the 0.38 detection threshold), so
    the second round RE-DETECTS and suppresses those frames under 0.55.

    isolation_only=True repairs ONLY isolated single-frame harsh bursts (true
    machine noise) and preserves multi-frame runs, which are sibilants/diction.
    Measured on the 60s HeartMuLa stem this halves the 5-8k diction loss vs the
    standard pass (iso -1.0 dB vs std1 -1.6 dB vs std2 -1.8 dB) while still
    removing the isolated burst frames. Human refs trigger the same zcr events,
    so the standard passes were eating real consonants.

    Dropout/level-drop recovery runs once on the first-round regions.
    """
    y, sr = sf.read(input_wav, always_2d=True)
    if y.shape[1] >= 2:
        mono = y.mean(axis=1)
    else:
        mono = y[:, 0]
    if mono.dtype != np.float64:
        mono = mono.astype(np.float64)
    duration_s = len(mono) / sr

    events = detect_events(input_wav, max_events_per_file=1000)
    regions = _events_to_regions(events, duration_s)
    if isolation_only:
        # keep structural regions (dropout/level drop) from the full detector,
        # but replace the harsh regions with the isolation-filtered subset
        structural = [r for r in regions if not r["types"] <= {"harsh_machine_noise"}]
        isolated = _isolated_harsh_regions(input_wav, duration_s)
        regions = structural + isolated
        events = [e for e in events if e["type"] != "harsh_machine_noise"]
        events += [{"time_seconds": r["start"], "type": "harsh_machine_noise", "severity": 0.0}
                   for r in isolated]
    steps: list[dict] = []
    repaired = mono.copy()
    round_tmp = output_wav.with_name(f".{output_wav.stem}_round.wav")

    # Each round re-detects on the current signal so frames that a previous
    # round pushed close to the threshold (but still above the 0.38 detector
    # floor) are found and suppressed again.
    round_regions = regions
    for _round in range(max(1, repair_rounds)):
        noise_regions = [r for r in round_regions if "harsh_machine_noise" in r["types"]]
        if not noise_regions:
            break
        import librosa

        S = librosa.stft(repaired, n_fft=N_FFT, hop_length=HOP, center=True)
        mask = _noise_mask(sr)
        for region in noise_regions:
            frame_t0 = max(0, int(region["start"] * sr / HOP) - 1)
            frame_t1 = min(S.shape[1], int(region["end"] * sr / HOP) + 2)
            S[:, frame_t0:frame_t1] *= mask[:, None]
            if _round == 0:
                steps.append({
                    "start": round(region["start"], 3),
                    "end": round(region["end"], 3),
                    "types": sorted(region["types"]),
                    "harsh_noise": {"status": "applied", "atten_db": round(20 * np.log10(NOISE_ATTEN), 1)},
                })
        repaired = librosa.istft(S, hop_length=HOP, center=True, length=len(mono))
        # Crossfade each noise region back into the previous round's signal so
        # edges stay clean without re-introducing what earlier rounds removed.
        previous = mono if _round == 0 else round_previous
        for region in noise_regions:
            lo = max(0, int(region["start"] * sr))
            hi = min(len(repaired), int(region["end"] * sr))
            if hi - lo < 128:
                continue
            ramp_len = max(1, int(0.01 * sr))
            ramp = min(ramp_len, (hi - lo) // 4)
            if ramp > 0:
                fade = np.ones(hi - lo)
                fade[:ramp] = np.linspace(0.0, 1.0, ramp)
                fade[-ramp:] = np.linspace(1.0, 0.0, ramp)
                repaired[lo:hi] = previous[lo:hi] * (1.0 - fade) + repaired[lo:hi] * fade
        round_previous = repaired.copy()
        # Re-detect on the current signal for the next round.
        if _round + 1 < max(1, repair_rounds):
            round_tmp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(round_tmp, np.stack([repaired, repaired], axis=1) if y.shape[1] >= 2
                     else repaired.reshape(-1, 1), sr, subtype="PCM_16")
            events = detect_events(round_tmp, max_events_per_file=1000)
            round_regions = _events_to_regions(events, duration_s)
    round_tmp.unlink(missing_ok=True)

    for region in regions:
        step = {"start": round(region["start"], 3), "end": round(region["end"], 3),
                "types": sorted(region["types"])}
        if "abrupt_level_drop" in region["types"]:
            repaired, result = _gain_recover(repaired, sr, region["start"], region["end"])
            step["level_drop"] = result
        if "possible_dropout" in region["types"]:
            repaired, result = _gain_recover(repaired, sr, region["start"], region["end"])
            step["dropout"] = result
        if "harsh_machine_noise" in region["types"]:
            continue  # already reported in the STFT pass
        steps.append(step)

    out = np.stack([repaired, repaired], axis=1) if y.shape[1] >= 2 else repaired.reshape(-1, 1)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, out, sr, subtype="PCM_16")
    return {
        "input": str(input_wav),
        "output": str(output_wav),
        "sample_rate": sr,
        "duration_s": round(duration_s, 3),
        "event_count": len(events),
        "region_count": len(regions),
        "noise_region_count": sum(1 for r in regions if "harsh_machine_noise" in r["types"]),
        "repair_rounds": max(1, repair_rounds),
        "steps": steps,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=1,
                        help="Repair passes (2 removes more harsh noise but eats sibilants).")
    parser.add_argument("--isolation-only", action="store_true",
                        help="Repair only isolated single-frame bursts; preserve sibilant runs (diction).")
    args = parser.parse_args()
    report = repair_vocals(args.input, args.output, repair_rounds=args.rounds,
                           isolation_only=args.isolation_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
