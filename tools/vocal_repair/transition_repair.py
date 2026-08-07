"""Surgical pitch-transition ("转音台阶") repair for AI vocal stems.

Detects three defect classes at note transitions (pitch step without glide,
true-silence gap, unpitched screech segment) and resynthesizes only the
detected windows with a WORLD portamento glide, leaving everything else
bit-exact. See docs/superpowers/specs/2026-08-07-transition-repair-design.md.
"""
from __future__ import annotations

import numpy as np

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
SCREECH_STD_ST = 2.5  # region voiced-frame st() spread marking erratic F0
SCREECH_JITTER_ST = 2.0  # region mean frame-to-frame st() jump marking erratic F0


def st_of(hz: float) -> float:
    return 12 * np.log2(max(hz, 1.0) / 440.0) + 69.0


def extract_f0(x: np.ndarray, sr: int) -> np.ndarray:
    """pyworld harvest F0, 5ms frame period (frame_s = HOP / sr)."""
    import pyworld as pw
    frame_period = HOP * 1000.0 / sr
    f0, _ = pw.harvest(x.astype(np.float64), sr, frame_period=frame_period,
                       f0_floor=F0_MIN, f0_ceil=F0_MAX)
    return f0


def _erratic(f0_region: np.ndarray) -> bool:
    """True when a region's voiced frames wander like harvest-locked noise.

    Real sung notes are flat to within ~1 st; harvest-locked noise jumps
    around by many semitones, which shows up either as high spread of st()
    across the region's voiced frames or as high mean frame-to-frame jitter.
    """
    sts = st_of(f0_region[f0_region > 0])
    if sts.size < 2:
        return False
    return (float(np.std(sts)) > SCREECH_STD_ST
            or float(np.mean(np.abs(np.diff(sts)))) > SCREECH_JITTER_ST)


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

    # Segment each voiced run into stable notes: a frame starts a new note
    # only when >= 2 *consecutive* frames deviate > 1 st from the running
    # median of the *current* segment, so single-frame harvest blips and a
    # lone off-pitch boundary frame cannot fragment a note. The running
    # median covers only the current segment's frames (seg_start..k).
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
        seg_start = 0  # offset into run of the current segment's first frame
        med = run[0]
        dev_count = 0
        for k in range(1, len(run)):
            if abs(st_of(run[k]) - st_of(med)) > 1.0:
                dev_count += 1
                if dev_count >= 2:
                    notes.append({"start": i + seg_start, "end": i + k,
                                  "f0": float(med)})
                    seg_start = k
                    med = run[k]
                    dev_count = 0
            else:
                dev_count = 0
                med = np.median(run[seg_start:k + 1])
        notes.append({"start": i + seg_start, "end": run_end, "f0": float(med)})
        i = run_end

    events: list[dict] = []
    j = 0
    while j < len(notes):
        a = notes[j]
        if (a["end"] - a["start"]) * frame_s < min_note_s:
            j += 1
            continue
        # The next candidate note must itself be long enough; short notes
        # between a and b (harvest-locked noise fragments, edge blips) are
        # part of the inter-note region, not transitions of their own.
        b = None
        k = j + 1
        while k < len(notes):
            if (notes[k]["end"] - notes[k]["start"]) * frame_s >= min_note_s:
                b = notes[k]
                break
            k += 1
        if b is None:
            break
        gap_s = (b["start"] - a["end"]) * frame_s
        jump = st_of(b["f0"]) - st_of(a["f0"])
        if abs(jump) < min_step_st:
            j += 1
            continue
        t0 = a["end"] * frame_s
        t1 = b["start"] * frame_s
        # enforce a minimum window (adjacent notes)
        if t1 - t0 < MIN_WINDOW_S:
            mid = (t0 + t1) / 2.0
            t0, t1 = mid - MIN_WINDOW_S / 2, mid + MIN_WINDOW_S / 2
        etype = "step"
        bridged = False
        if x is not None:
            i0 = int(t0 * sr)
            i1 = int(t1 * sr)
            if i1 > i0:
                w = int(0.025 * sr)
                rms = []
                for m in range(i0, i1, w // 2):
                    m2 = min(m + w, i1)  # never overrun the inter-note window
                    if m2 <= m:
                        continue
                    rms.append(20 * np.log10(np.sqrt(np.mean(x[m:m2] ** 2)) + 1e-9))
                rms_db = np.asarray(rms)
                if rms_db.size and rms_db.max() < silence_db:
                    etype = "silence_gap"
                elif rms_db.size and rms_db.max() >= screech_rms_db:
                    bridged = True  # energy bridges the gap (noise segment)
                    mid0, mid1 = a["end"], b["start"]
                    if mid1 > mid0 and (mid1 - mid0) * frame_s >= screech_min_s:
                        region = f0[mid0:mid1]
                        if np.any(region == 0) or _erratic(region):
                            etype = "screech"
        # The glide budget gates clean step/silence gaps only; a
        # noise-bridged gap is a screech defect regardless of length.
        if gap_s > glide_max_s and not bridged:
            j += 1
            continue
        events.append({
            "type": etype, "t0": float(t0), "t1": float(t1),
            "f_a": float(a["f0"]), "f_b": float(b["f0"]),
            "jump_st": float(jump), "flags": [],
        })
        j += 1
    return events
