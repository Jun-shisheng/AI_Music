"""Extend a finished mix to a target duration with instrumental intro/outro loops.

The vocal and its original mix are untouched inside the song body. Only the
instrumental stem is used for the added intro and outro, with short crossfades
and a final fade-out. This is an arrangement edit, not audio repair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def xfade(a: np.ndarray, b: np.ndarray, samples: int) -> np.ndarray:
    samples = min(samples, len(a), len(b))
    if samples <= 0:
        return np.concatenate([a, b], axis=0)
    w = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    head = a[:-samples] if samples < len(a) else a[:0]
    return np.concatenate([head, a[-samples:] * (1 - w) + b[:samples] * w, b[samples:]], axis=0)


def trim_tail(data: np.ndarray, rate: int, threshold: float = 0.01) -> np.ndarray:
    hop = max(1, rate // 20)
    mono = np.abs(data).mean(axis=1)
    frames = len(mono) // hop
    levels = np.sqrt(np.mean(mono[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    active = np.where(levels > threshold)[0]
    if not len(active):
        return data
    end = min(len(data), (int(active[-1]) + 1) * hop + int(0.5 * rate))
    return data[:end]


def loop_fill(segment: np.ndarray, length: int, fade: int) -> np.ndarray:
    out = segment.copy()
    while len(out) < length:
        out = xfade(out, segment, fade)
    return out[:length]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--instrumental", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-seconds", type=float, default=210.0)
    parser.add_argument("--intro-bars", type=float, default=4.0)
    parser.add_argument("--loop-bars", type=float, default=4.0)
    parser.add_argument("--fade-out-seconds", type=float, default=4.0)
    parser.add_argument("--crossfade-ms", type=float, default=250)
    parser.add_argument("--loop-crossfade-ms", type=float, default=100)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    mix, mix_rate = sf.read(str(args.mix), dtype="float64", always_2d=True)
    inst, inst_rate = sf.read(str(args.instrumental), dtype="float64", always_2d=True)
    if mix_rate != 44100 or inst_rate != 44100:
        raise SystemExit("Both inputs must be 44.1 kHz")
    if abs(len(mix) - len(inst)) > mix_rate // 100:
        raise SystemExit("Mix and instrumental timelines differ")

    body = trim_tail(mix, mix_rate)
    current = len(body) / mix_rate

    mono, sr = librosa.load(str(args.instrumental), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, units="time")
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    beats = np.asarray(beats, dtype=float)
    beat = float(np.median(np.diff(beats))) if len(beats) > 4 else 60.0 / tempo
    bar = 4 * beat

    # End the body on a detected beat so the outro can start on the same phase.
    if len(beats):
        prior = beats[beats <= current + 0.02]
        if len(prior) and current - float(prior[-1]) < 1.0:
            body = body[: int(round(float(prior[-1]) * mix_rate))]
            current = len(body) / mix_rate
    inst = inst[: len(body)]
    extra = max(0.0, args.target_seconds - current)
    if extra <= 0:
        raise SystemExit(f"Mix is already {current:.3f}s, no extension needed")

    intro_seconds = min(extra * 0.35, args.intro_bars * bar)
    intro_seconds = max(0.0, intro_seconds)
    intro_beats = int(round(args.intro_bars)) * 4
    if len(beats) > intro_beats and beats[intro_beats] < extra * 0.45:
        intro_seconds = float(beats[intro_beats])
    outro_seconds = extra - intro_seconds
    intro_n = int(round(intro_seconds * mix_rate))
    outro_n = int(round(outro_seconds * mix_rate))

    intro_block = inst[:intro_n].copy()
    fade_in = min(int(0.030 * mix_rate), len(intro_block))
    if fade_in:
        intro_block[:fade_in] *= np.linspace(0, 1, fade_in, endpoint=False)[:, None]

    # Choose the loudest bar-aligned 4-bar window in the last 45% of the song,
    # excluding the final 4 seconds, as the outro loop unit.
    loop_beats = int(round(args.loop_bars)) * 4
    search_time = current * 0.55
    search_end_time = max(search_time + loop_beats * beat, current - 4.0)
    candidates = []
    if len(beats) > loop_beats:
        for idx in range(0, len(beats) - loop_beats, 4):
            start_t = float(beats[idx])
            end_t = float(beats[idx + loop_beats])
            if start_t < search_time or end_t > search_end_time:
                continue
            a, b = int(round(start_t * mix_rate)), int(round(end_t * mix_rate))
            seg = inst[a:b]
            if len(seg) < int(0.5 * mix_rate):
                continue
            candidates.append((float(np.sqrt(np.mean(seg ** 2))), a, b, start_t))
    if candidates:
        best_rms, best_start, loop_end, loop_start_t = max(candidates)
        loop_unit = inst[best_start:loop_end].copy()
        loop_len = len(loop_unit)
    else:
        loop_len = int(round(args.loop_bars * bar * mix_rate))
        best_start = int(len(inst) * 0.7)
        loop_unit = inst[best_start:best_start + loop_len].copy()
        loop_start_t = best_start / mix_rate
        best_rms = float(np.sqrt(np.mean(loop_unit ** 2)))

    outro_block = loop_fill(loop_unit, outro_n, int(args.loop_crossfade_ms / 1000 * mix_rate))
    fade_out = min(int(args.fade_out_seconds * mix_rate), len(outro_block))
    if fade_out:
        outro_block[-fade_out:] *= np.linspace(1, 0, fade_out, endpoint=False)[:, None]

    xf = int(args.crossfade_ms / 1000 * mix_rate)
    extended = body
    if len(intro_block):
        extended = xfade(intro_block, extended, xf)
    if len(outro_block):
        extended = xfade(extended, outro_block, xf)

    peak = float(np.max(np.abs(extended)))
    gain = min(1.0, 0.95 / peak) if peak else 1.0
    extended = extended * gain
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), extended, mix_rate, subtype="PCM_16")

    report = {
        "schema": 1, "kind": "instrumental_length_extension",
        "source_mix": str(args.mix), "instrumental": str(args.instrumental),
        "body_seconds": round(current, 6), "target_seconds": args.target_seconds,
        "output_seconds": round(len(extended) / mix_rate, 6),
        "tempo_bpm": round(tempo, 3), "beat_seconds": round(beat, 6), "bar_seconds": round(bar, 6),
        "intro_seconds": round(len(intro_block) / mix_rate, 6),
        "outro_seconds": round(len(outro_block) / mix_rate, 6),
        "outro_loop_start_seconds": round(best_start / mix_rate, 6),
        "outro_loop_rms": round(best_rms, 6),
        "crossfade_ms": args.crossfade_ms, "loop_crossfade_ms": args.loop_crossfade_ms,
        "fade_out_seconds": args.fade_out_seconds,
        "global_gain_db": round(float(20 * np.log10(gain)), 6),
        "vocal_processed": False, "time_varying_processing": False,
        "note": "Only the instrumental stem is looped for intro/outro; the song body is unchanged.",
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
