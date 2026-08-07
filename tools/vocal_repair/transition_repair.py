"""Surgical pitch-transition ("转音台阶") repair for AI vocal stems.

Detects three defect classes at note transitions (pitch step without glide,
true-silence gap, unpitched screech segment) and resynthesizes only the
detected windows with a WORLD portamento glide, leaving everything else
bit-exact. See docs/superpowers/specs/2026-08-07-transition-repair-design.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "vocal_repair"))

HOP = 256
N_FFT = 2048
F0_MIN = 90.0
F0_MAX = 900.0
MIN_NOTE_S = 0.15
GLIDE_MAX_S = 0.30
MIN_STEP_ST = 2.0
SILENCE_DB = -60.0
SCREECH_MIN_S = 0.06
SCREECH_RMS_DB = -45.0
MIN_WINDOW_S = 0.10


def st_of(hz: float) -> float:
    return 12 * np.log2(max(hz, 1.0) / 440.0) + 69.0


def extract_f0(x: np.ndarray, sr: int) -> np.ndarray:
    """pyworld harvest F0, 5ms frame period (frame_s = HOP / sr)."""
    import pyworld as pw
    frame_period = HOP * 1000.0 / sr
    f0, _ = pw.harvest(x.astype(np.float64), sr, frame_period=frame_period,
                       f0_floor=F0_MIN, f0_ceil=F0_MAX)
    return f0


def detect_transition_events(f0: np.ndarray, sr: int,
                             min_step_st: float = MIN_STEP_ST,
                             min_note_s: float = MIN_NOTE_S,
                             glide_max_s: float = GLIDE_MAX_S,
                             silence_db: float = SILENCE_DB,
                             screech_min_s: float = SCREECH_MIN_S,
                             screech_rms_db: float = SCREECH_RMS_DB,
                             x: np.ndarray | None = None) -> list[dict]:
    """Detect transition defect events.

    Requires `x` (the mono audio) for silence/screech RMS checks; without it
    only `step` events are detected.
    """
    frame_s = HOP / sr
    n = len(f0)
    voiced = f0 > 0

    # segment voiced runs into stable notes (±1 st around running median)
    notes: list[dict] = []  # {start_frame, end_frame, f0}
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        run_end = i
        while run_end < n and voiced[run_end]:
            run_end += 1
        run = f0[i:run_end]
        seg_start = i
        med = run[0]
        for k in range(1, len(run)):
            if abs(st_of(run[k]) - st_of(med)) > 1.0:
                notes.append({"start": seg_start, "end": i + k, "f0": float(med)})
                seg_start = i + k
                med = run[k]
            else:
                med = np.median(run[: k + 1])
        notes.append({"start": seg_start, "end": run_end, "f0": float(med)})
        i = run_end

    events: list[dict] = []
    for a, b in zip(notes[:-1], notes[1:]):
        dur_a = (b["start"] - a["start"]) * frame_s
        dur_b = (b["end"] - b["start"]) * frame_s
        if dur_a < min_note_s or dur_b < min_note_s:
            continue
        gap_s = (b["start"] - a["end"]) * frame_s
        if gap_s > glide_max_s:
            continue
        jump = st_of(b["f0"]) - st_of(a["f0"])
        if abs(jump) < min_step_st:
            continue
        t0 = a["end"] * frame_s
        t1 = b["start"] * frame_s
        # enforce a minimum window (adjacent notes)
        if t1 - t0 < MIN_WINDOW_S:
            mid = (t0 + t1) / 2.0
            t0, t1 = mid - MIN_WINDOW_S / 2, mid + MIN_WINDOW_S / 2
        etype = "step"
        if x is not None:
            i0 = int(t0 * sr)
            i1 = int(t1 * sr)
            if i1 > i0:
                w = int(0.025 * sr)
                rms_db = np.array([
                    20 * np.log10(np.sqrt(np.mean(x[j:j + w] ** 2)) + 1e-9)
                    for j in range(i0, i1, w // 2)
                ])
                if rms_db.size and rms_db.max() < silence_db:
                    etype = "silence_gap"
                elif rms_db.size and rms_db.max() >= screech_rms_db:
                    # check for F0-unvoiced energy between the notes
                    mid0, mid1 = a["end"], b["start"]
                    if mid1 > mid0 and np.any(f0[mid0:mid1] == 0):
                        seg_s = (mid1 - mid0) * frame_s
                        if seg_s >= screech_min_s:
                            etype = "screech"
        events.append({
            "type": etype, "t0": float(t0), "t1": float(t1),
            "f_a": float(a["f0"]), "f_b": float(b["f0"]),
            "jump_st": float(jump), "flags": [],
        })
    return events
