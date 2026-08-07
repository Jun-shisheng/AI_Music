# Transition Repair (转音修复) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix "转音台阶" (steppy pitch transitions) on AI vocal stems by surgically resynthesizing only detected transition windows with WORLD glide synthesis, leaving everything else bit-exact.

**Architecture:** Three isolated units in one module (`transition_repair.py`): (1) event detector using pyworld F0 (detects pitch steps, true-silence gaps, screech segments), (2) glide F0 builder producing natural portamento trajectories, (3) WORLD resynthesis + crossfaded blending into the original. The module integrates into the existing humanize chain (after noise_floor, before de_buzz) in both `quality_rebuild_agent.py` and the FastAPI `queue.py` hook, with the same listening-first preference and fallback semantics as the humanize step.

**Tech Stack:** Python 3.11 (tools/vocal_repair/.venv), numpy, scipy, soundfile, pyworld (0.3.5, API: `harvest`/`cheaptrick`/`d4c`/`synthesize`), ffmpeg. Tests run with pytest via `tools/vocal_repair/.venv/Scripts/python.exe -m pytest`.

**Spec:** `docs/superpowers/specs/2026-08-07-transition-repair-design.md`

---

## File Structure

- Create: `tools/vocal_repair/transition_repair.py` — detector + glide builder + WORLD blend, plus CLI
- Test: `tools/vocal_repair/tests/test_transition_repair.py` — synthetic unit tests
- Modify: `tools/vocal_repair/quality_rebuild_agent.py` — add transition repair as a candidate step
- Modify: `services/api/app/services/queue.py` — add VOCAL_TRANSITION_REPAIR hook
- Modify: `services/api/app/config.py` — add `vocal_transition_repair_enabled` setting
- Modify: `services/api/app/prompts/arrange_system.txt` — not needed (no prompt change)

## Environment & Commands

All Python runs use: `tools/vocal_repair/.venv/Scripts/python.exe` (contains pyworld, numpy, scipy, soundfile, librosa, pytest 9.1.1).

All pytest runs: `tools/vocal_repair/.venv/Scripts/python.exe -m pytest <path> -v`

Spec test data: `outputs/transition_work/02_HeartMuLa_60s_moonlit/demucs/htdemucs/02_HeartMuLa_60s_moonlit/vocals.wav` (60s vocal stem, 44.1kHz, has known transition defects at ~39.2s silence gap, ~49.5s screech, ~50.9s step).

---

### Task 1: Event detector with tests

**Files:**
- Create: `tools/vocal_repair/transition_repair.py`
- Create: `tools/vocal_repair/tests/test_transition_repair.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_transition_repair.py
"""Unit tests for transition_repair event detection."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transition_repair import detect_transition_events, extract_f0  # noqa: E402

SR = 44100
HOP = 256
FRAME_S = HOP / SR
ST = lambda hz: 12 * np.log2(max(hz, 1) / 440.0) + 69.0


def make_audio_with_step(tmp_path, gap_s=0.2, jump_st=6.0, f0_a=220.0):
    """Two stable notes with a hard pitch step and optional silent gap."""
    note_dur = 0.4
    sr = SR
    n1 = int(note_dur * sr)
    n2 = int(note_dur * sr)
    gap = int(gap_s * sr)
    t1 = np.arange(n1) / sr
    t2 = np.arange(n2) / sr
    f0_b = f0_a * (2 ** (jump_st / 12))
    sig = np.concatenate([
        0.6 * np.sin(2 * np.pi * f0_a * t1),
        np.zeros(gap),
        0.6 * np.sin(2 * np.pi * f0_b * t2),
    ])
    return sig.astype(np.float64), sr


def test_detects_pitch_step_with_silent_gap(tmp_path):
    sig, sr = make_audio_with_step(tmp_path)
    f0 = extract_f0(sig, sr)
    events = detect_transition_events(f0, sr)
    steps = [e for e in events if e["type"] == "step"]
    assert len(steps) >= 1
    e = steps[0]
    assert abs(ST(e["f_a"]) - ST(220.0)) < 1.5
    assert abs(ST(e["f_b"]) - ST(220.0 * 2 ** (6 / 12))) < 1.5
    # window should cover the gap
    assert e["t1"] - e["t0"] >= gap_s - 0.05


def test_detects_step_with_adjacent_notes_no_gap(tmp_path):
    """Notes touching (no silence) still produce a step event with min window."""
    sig, sr = make_audio_with_step(tmp_path, gap_s=0.0)
    f0 = extract_f0(sig, sr)
    events = detect_transition_events(f0, sr)
    steps = [e for e in events if e["type"] == "step"]
    assert len(steps) >= 1
    # minimum window enforced
    assert steps[0]["t1"] - steps[0]["t0"] >= 0.08


def test_no_events_on_steady_sine(tmp_path):
    t = np.arange(SR) / SR
    sig = 0.6 * np.sin(2 * np.pi * 220 * t)
    f0 = extract_f0(sig, SR)
    events = detect_transition_events(f0, SR)
    assert len(events) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `tools/vocal_repair/.venv/Scripts/python.exe -m pytest tools/vocal_repair/tests/test_transition_repair.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'transition_repair'`

- [ ] **Step 3: Implement the detector**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `tools/vocal_repair/.venv/Scripts/python.exe -m pytest tools/vocal_repair/tests/test_transition_repair.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/vocal_repair/transition_repair.py tools/vocal_repair/tests/test_transition_repair.py
git commit -m "feat: transition event detector for vocal stems"
```

---

### Task 2: Glide F0 builder + WORLD blend with tests

**Files:**
- Modify: `tools/vocal_repair/transition_repair.py` (add `build_glide_f0`, `repair_transitions`, `analyze_world`, `synthesize_window`)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_transition_repair.py
import soundfile as sf
from transition_repair import build_glide_f0, repair_transitions


def test_glide_f0_is_monotonic_and_ends_at_target(tmp_path):
    # f_a=220Hz, f_b=440Hz over 200ms
    t0, t1 = 0.0, 0.2
    f0 = build_glide_f0(220.0, 440.0, t0, t1, SR)
    assert abs(f0[0] - 220.0) < 2.0
    assert abs(f0[-1] - 440.0) < 2.0
    # log-frequency glide: monotonic in st
    st = [st_of(f) for f in f0]
    assert all(st[i] <= st[i + 1] + 1e-6 for i in range(len(st) - 1))
    assert f0[-1] > f0[0]


def test_repair_step_preserves_outside_samples_bit_exact(tmp_path):
    """Repairing a step keeps everything outside the event window bit-exact."""
    sig, sr = make_audio_with_step(tmp_path, gap_s=0.15, jump_st=6.0)
    x = sig
    repaired, report = repair_transitions(x, sr)
    assert len(report["events"]) >= 1
    e = report["events"][0]
    i0 = int(e["t0"] * sr)
    i1 = int(e["t1"] * sr)
    np.testing.assert_array_equal(repaired[:i0], x[:i0])
    np.testing.assert_array_equal(repaired[i1:], x[i1:])
    # the gap region should now contain audio (not silence)
    gap = repaired[i0:i1]
    assert np.sqrt(np.mean(gap ** 2)) > 1e-4


def test_repair_idempotent_on_clean_signal(tmp_path):
    t = np.arange(SR) / SR
    sig = 0.6 * np.sin(2 * np.pi * 220 * t)
    repaired, report = repair_transitions(sig, SR)
    assert len(report["events"]) == 0
    np.testing.assert_array_equal(repaired, sig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `tools/vocal_repair/.venv/Scripts/python.exe -m pytest tools/vocal_repair/tests/test_transition_repair.py -k "glide or repair" -v`
Expected: FAIL with `ImportError: cannot import name 'build_glide_f0'`

- [ ] **Step 3: Implement the glide builder + WORLD blend**

```python
# append to transition_repair.py

def build_glide_f0(f_a: float, f_b: float, t0: float, t1: float,
                   sr: int, hold_frac: float = 0.3,
                   overshoot_st: float = 0.4) -> np.ndarray:
    """Exponential glide F0 over [t0, t1] with natural shape.

    Holds f_a for the first hold_frac of the window, glides log-linearly,
    overshoots overshoot_st semitones at ~80%, settles to f_b at the end.
    Returns per-frame F0 at HOP spacing, indexed from t0.
    """
    n_frames = max(1, int(round((t1 - t0) * sr / HOP)))
    k = np.arange(n_frames) / max(n_frames - 1, 1)
    st_a, st_b = st_of(f_a), st_of(f_b)
    # hold + exponential glide + overshoot, clamped to monotonic-ish settle
    hold_end = hold_frac
    glide_k = np.clip((k - hold_end) / (1 - hold_end), 0, 1)
    st_glide = st_a + (st_b - st_a) * glide_k
    # overshoot at ~80% of the *window* (after hold), then settle
    overshoot_pos = hold_end + 0.8 * (1 - hold_end)
    st_shape = np.where(
        k < overshoot_pos,
        st_glide,
        st_b + overshoot_st * (1 - (k - overshoot_pos) / (1 - overshoot_pos))
    )
    st_shape = np.clip(st_shape, st_a - 1.0, st_b + overshoot_st + 1.0)
    f0 = 440.0 * 2 ** ((st_shape - 69.0) / 12.0)
    # edges converge exactly to the notes
    f0[0] = f_a
    f0[-1] = f_b
    return f0


def analyze_world(x: np.ndarray, sr: int) -> dict:
    """Full-stem WORLD analysis: f0, spectral envelope, aperiodicity."""
    import pyworld as pw
    frame_period = HOP * 1000.0 / sr
    f0, _ = pw.harvest(x.astype(np.float64), sr, frame_period=frame_period,
                       f0_floor=F0_MIN, f0_ceil=F0_MAX)
    sp = pw.cheaptrick(x.astype(np.float64), f0, sr, frame_period=frame_period)
    ap = pw.d4c(x.astype(np.float64), f0, sr, frame_period=frame_period)
    return {"f0": f0, "sp": sp, "ap": ap}


def synthesize_window(x: np.ndarray, sr: int, world: dict, ev: dict,
                      glide: np.ndarray) -> np.ndarray:
    """WORLD-synthesize the glide for one event window, return the audio."""
    import pyworld as pw
    frame_period = HOP * 1000.0 / sr
    t0, t1 = ev["t0"], ev["t1"]
    i0, i1 = int(t0 * sr), int(t1 * sr)
    i0 = max(0, i0); i1 = min(len(x), i1)
    if i1 <= i0:
        return np.zeros(0)
    # frame indices aligned to HOP spacing (frame_period = HOP/sr * 1000)
    n_frames = int(round((t1 - t0) / (HOP / sr)))
    fr0 = int(round(t0 / (HOP / sr)))
    fr1 = min(fr0 + n_frames, len(world["sp"]))
    f0w = np.zeros(n_frames)
    f0w[:] = glide[:n_frames]
    # envelope: silence_gap interpolates from both notes; else use window's own
    if ev["type"] == "silence_gap":
        sp_a = world["sp"][fr0 - 1] if fr0 - 1 >= 0 else world["sp"][0]
        sp_b = world["sp"][fr1] if fr1 < len(world["sp"]) else world["sp"][-1]
        spw = np.stack([
            sp_a + (sp_b - sp_a) * (k / max(n_frames - 1, 1))
            for k in range(n_frames)
        ])
    else:
        spw = world["sp"][fr0:fr0 + n_frames] if fr0 + n_frames <= len(world["sp"]) \
            else np.tile(world["sp"][fr0], (n_frames, 1))
    # aperiodicity: screech blends toward surrounding notes
    if ev["type"] == "screech":
        ap_a = world["ap"][fr0 - 1] if fr0 - 1 >= 0 else world["ap"][0]
        ap_b = world["ap"][fr1] if fr1 < len(world["ap"]) else world["ap"][-1]
        apw = np.stack([
            ap_a + (ap_b - ap_a) * (k / max(n_frames - 1, 1))
            for k in range(n_frames)
        ])
    else:
        apw = world["ap"][fr0:fr0 + n_frames] if fr0 + n_frames <= len(world["ap"]) \
            else np.tile(world["ap"][fr0], (n_frames, 1))
    y = pw.synthesize(f0w, spw, apw, sr, frame_period)
    # pad to window length (pyworld returns slightly shorter)
    y = np.pad(y, (0, max(0, (i1 - i0) - len(y))))
    return y


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int, fade_s: float = 0.030) -> np.ndarray:
    """Overlap-add b (the repaired window) over a with raised-cosine fades."""
    nf = min(int(fade_s * sr), len(a), len(b))
    if nf <= 0:
        return b
    out = a.copy()
    ramp = np.linspace(0, 1, nf)
    out[:nf] = a[:nf] * (1 - ramp) + b[:nf] * ramp
    out[-nf:] = a[-nf:] * (1 - ramp[::-1]) + b[-nf:] * ramp[::-1]
    return out


def repair_transitions(x: np.ndarray, sr: int,
                       max_gap_s: float = 0.30) -> tuple[np.ndarray, dict]:
    """Repair all detected transition events on a mono vocal stem.

    Returns (repaired_audio, report). Outside event windows is bit-exact.
    Long gaps (> max_gap_s) are flagged in the report, not repaired.
    """
    f0 = extract_f0(x, sr)
    events = detect_transition_events(f0, sr, x=x)
    report = {"events": events, "repaired": []}
    if not events:
        return x.copy(), report
    world = analyze_world(x, sr)
    out = x.copy()
    for ev in events:
        if ev["t1"] - ev["t0"] > max_gap_s:
            ev["flags"].append("long_gap_skipped")
            report["repaired"].append({"type": ev["type"], "t0": ev["t0"],
                                       "t1": ev["t1"], "skipped": True})
            continue
        try:
            glide = build_glide_f0(ev["f_a"], ev["f_b"], ev["t0"], ev["t1"], sr)
            win = synthesize_window(x, sr, world, ev, glide)
            i0, i1 = int(ev["t0"] * sr), int(ev["t1"] * sr)
            i1 = min(i1, len(out))
            if i1 > i0 and len(win) == i1 - i0:
                seg = out[i0:i1]
                out[i0:i1] = _crossfade(seg, win, sr)
                report["repaired"].append({"type": ev["type"], "t0": ev["t0"],
                                           "t1": ev["t1"], "skipped": False})
            else:
                ev["flags"].append("length_mismatch")
                report["repaired"].append({"type": ev["type"], "t0": ev["t0"],
                                           "t1": ev["t1"], "skipped": True})
        except Exception as exc:  # noqa: BLE001 - per-event best effort
            ev["flags"].append(f"error:{type(exc).__name__}")
            report["repaired"].append({"type": ev["type"], "t0": ev["t0"],
                                       "t1": ev["t1"], "skipped": True})
    return out, report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `tools/vocal_repair/.venv/Scripts/python.exe -m pytest tools/vocal_repair/tests/test_transition_repair.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Commit**

```bash
git add tools/vocal_repair/transition_repair.py tools/vocal_repair/tests/test_transition_repair.py
git commit -m "feat: glide F0 builder and WORLD transition repair"
```

---

### Task 3: CLI entry + real-file sanity check

**Files:**
- Modify: `tools/vocal_repair/transition_repair.py` (add `main()`)

The CLI must handle two modes, mirroring `humanize_vocal.py`:
- `transition_repair.py <vocals.wav> <out.wav> [--report ...]`: repair a raw vocal stem (no demucs, no remix).
- `transition_repair.py <song.wav> --output <out.wav> [--work-dir ...]`: full pipeline (demucs separate -> frame repair -> noise floor -> transition repair -> remix), matching how `queue.py` will invoke it on a mixed song file.

- [ ] **Step 1: Add CLI**

```python
# append to transition_repair.py

def _analyze_file(x: np.ndarray, sr: int) -> dict:
    """Reuse extract_f0 + detect for report output."""
    f0 = extract_f0(x, sr)
    events = detect_transition_events(f0, sr, x=x)
    return {"sr": sr, "events": events}


def _repair_wav(input_wav: Path, output: Path, max_gap_s: float,
                report_path: Path | None) -> dict:
    y, sr = sf.read(input_wav, always_2d=True)
    x = y.mean(axis=1).astype(np.float64) if y.ndim > 1 else y.astype(np.float64)
    out, report = repair_transitions(x, sr, max_gap_s=max_gap_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    if y.ndim > 1:
        final = np.stack([out, out], axis=1)
    else:
        final = out
    final = np.clip(final, -1, 1)
    sf.write(output, final, sr, subtype="PCM_16")
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    from stem_pipeline import remix, separate_with_demucs

    parser = argparse.ArgumentParser(description="Repair pitch-transition defects on a vocal stem")
    parser.add_argument("input", type=Path, help="vocal stem (raw mode) or song WAV (full mode)")
    parser.add_argument("output", type=Path, help="output WAV path")
    parser.add_argument("--max-gap-ms", type=float, default=300.0)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "outputs" / "transition_work")
    args = parser.parse_args()

    # full mode: --work-dir given AND input looks like a full mix (stereo, has instrumental)
    if args.work_dir is not None and args.output.suffix.lower() == ".wav" and args.input.suffix.lower() == ".wav":
        # check: if the file is a full mix (long stereo with accompaniment), separate it
        y, sr = sf.read(args.input, always_2d=True)
        if y.shape[1] >= 2 and len(y) / sr > 10.0:
            song_work = args.work_dir / args.input.stem
            vocals, instrumental = separate_with_demucs(args.input, song_work)
            # frame repair + noise floor first (same as humanize_stem order)
            from frame_repair import repair_vocals
            from noise_floor_reducer import reduce_noise_floor
            reduced = song_work / "vocals_noise_reduced.wav"
            reduce_noise_floor(vocals, reduced)
            repaired = song_work / "vocals_frame_repaired.wav"
            repair_vocals(reduced, repaired)
            # transition repair on the repaired vocal stem
            y2, sr2 = sf.read(repaired, always_2d=True)
            x2 = y2.mean(axis=1).astype(np.float64) if y2.ndim > 1 else y2.astype(np.float64)
            out2, report = repair_transitions(x2, sr2, max_gap_s=args.max_gap_ms / 1000.0)
            final2 = np.clip(out2, -1, 1)
            if y2.ndim > 1:
                final2 = np.stack([final2, final2], axis=1)
            staged = song_work / "vocals_transition_repaired.wav"
            sf.write(staged, final2, sr2, subtype="PCM_16")
            remix(instrumental, staged, args.output)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"full pipeline: repaired {len(report['repaired'])} events -> {args.output}")
            return 0

    # raw mode: single vocal stem
    report = _repair_wav(args.input, args.output, args.max_gap_ms / 1000.0, args.report)
    print(f"repaired {len(report['repaired'])} events -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run on the real HeartMuLa vocal stem**

Run:
```bash
tools/vocal_repair/.venv/Scripts/python.exe tools/vocal_repair/transition_repair.py \
  outputs/transition_work/02_HeartMuLa_60s_moonlit/demucs/htdemucs/02_HeartMuLa_60s_moonlit/vocals.wav \
  outputs/transition_work/vocals_transition_repaired.wav --report outputs/transition_work/transition_report.json
```
Expected: prints "repaired N events" (N >= 1). Inspect `transition_report.json`: silence_gap at ~39.2s should be flagged `long_gap_skipped` (1.3s > 300ms) unless `--max-gap-ms` is raised; step at ~50.9s and screech at ~49.5s should be repaired.

- [ ] **Step 3: Verify bit-exactness outside windows on the real file**

Run:
```bash
tools/vocal_repair/.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'tools/vocal_repair')
import numpy as np, soundfile as sf
from transition_repair import extract_f0, detect_transition_events
x, sr = sf.read('outputs/transition_work/02_HeartMuLa_60s_moonlit/demucs/htdemucs/02_HeartMuLa_60s_moonlit/vocals.wav', always_2d=True)
x = x.mean(axis=1)
r, sr2 = sf.read('outputs/transition_work/vocals_transition_repaired.wav', always_2d=True)
r = r.mean(axis=1)
f0 = extract_f0(x, sr)
evs = detect_transition_events(f0, sr, x=x)
outside = np.ones(len(x), dtype=bool)
for e in evs:
    i0, i1 = int(e['t0']*sr), int(e['t1']*sr)
    outside[i0:i1] = False
print('samples outside windows identical:', np.array_equal(x[outside], r[outside]))
"
```
Expected: `samples outside windows identical: True`

- [ ] **Step 4: Commit**

```bash
git add tools/vocal_repair/transition_repair.py
git commit -m "feat: transition repair CLI and real-file verification"
```

---

### Task 4: Prototype A/B listening clips

**Files:**
- none (produces outputs)

- [ ] **Step 1: Render repaired mix and extract the 4 transition clips**

Run (in the vocal_repair venv):
```bash
tools/vocal_repair/.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'tools/vocal_repair')
import subprocess, numpy as np, soundfile as sf
from transition_repair import repair_transitions
from stem_pipeline import remix
from pathlib import Path

work = Path('outputs/transition_work')
vocals = work/'demucs'/'htdemucs'/'02_HeartMuLa_60s_moonlit'/'vocals.wav'
inst = work/'demucs'/'htdemucs'/'02_HeartMuLa_60s_moonlit'/'no_vocals.wav'
y, sr = sf.read(vocals, always_2d=True)
x = y.mean(axis=1).astype(np.float64)
out, report = repair_transitions(x, sr)
# write repaired vocal stem
final = np.stack([out, out], axis=1)
sf.write(work/'vocals_transition_repaired.wav', final, sr, subtype='PCM_16')
# remix with instrumental
remix(inst, work/'vocals_transition_repaired.wav', work/'mix_transition_repaired.wav')
print('events:', [(e['type'], round(e['t0'],2), round(e['t1'],2)) for e in report['events']])
"
```
Then extract clips (ffmpeg) for A/B:
```bash
for name_center in "t40 38.0" "t50 49.0" "t27 26.0" "t_end 55.0"; do
  set -- $name_center
  ffmpeg -y -v error -ss $2 -t 4.5 -i outputs/transition_work/mix_transition_repaired.wav outputs/transition_listen/${1}_repaired.wav
  ffmpeg -y -v error -ss $2 -t 4.5 -i outputs/ab_compare/02_HeartMuLa_60s_moonlit.wav outputs/transition_listen/${1}_orig.wav
done
```

- [ ] **Step 2: User A/B listening**

Ask the user to listen to `outputs/transition_listen/*_orig.wav` vs `*_repaired.wav` (4 pairs) plus the full repaired mix, and confirm:
- transitions sound glided (not steppy)
- no new artifacts (clicks, wobble, metallic) at the repaired windows
- overall vocal timbre unchanged

**Gate:** proceed to Task 5 only on user approval. If the user rejects (e.g. WORLD texture too thin at female high notes, or the 39s 1.3s gap still feels wrong), iterate on glide parameters (hold_frac, overshoot, max_gap_ms) or fall back to flagging long gaps as known-limitation.

- [ ] **Step 3: Commit (prototype only if approved)**

```bash
git add outputs/transition_work/transition_report.json
git commit -m "chore: transition repair prototype A/B evidence"
```
(If the user rejects, commit nothing and adjust.)

---

### Task 5: Integrate into quality_rebuild_agent.py

**Files:**
- Modify: `tools/vocal_repair/quality_rebuild_agent.py`

- [ ] **Step 1: Add transition repair candidate + preference**

Add import near line 14 and the candidate step in `rebuild()` (after humanize, before candidates list):

```python
from transition_repair import repair_transitions as repair_transitions_fn
```

In `rebuild()` after the humanize block (around line 198):

```python
# Transition repair: surgical WORLD glide synthesis at pitch-step /
# silence-gap / screech windows. Chosen when it beats the humanize chain on
# the listening-first preference (never regresses gate/structure/dropout).
transitioned = song_work / "vocals_transition_repaired.wav"
transition_error = ""
try:
    y_tr, sr_tr = sf.read(vocals, always_2d=True)
    x_tr = y_tr.mean(axis=1).astype(np.float64) if y_tr.ndim > 1 else y_tr.astype(np.float64)
    out_tr, _ = repair_transitions_fn(x_tr, sr_tr)
    final_tr = np.clip(out_tr, -1, 1)
    if y_tr.ndim > 1:
        final_tr = np.stack([final_tr, final_tr], axis=1)
    sf.write(transitioned, final_tr, sr_tr, subtype="PCM_16")
    transition_after = measure(transitioned, wav)
except Exception as exc:  # noqa: BLE001 - transition repair is best-effort
    transition_after = before
    transition_error = f"{type(exc).__name__}: {exc}"
```

Add to `candidates` (after humanize, line ~205):
```python
if not transition_error:
    candidates.append(("transition", transition_after, transitioned))
```

Add listening-first preference (after the humanize preference block, ~line 218):
```python
transition_entry = next((c for c in candidates if c[0] == "transition"), None)
if transition_entry is not None and humanize_preferred(before, transition_entry[1]):
    chosen = transition_entry
```

Update the `evaluated.append(...)` dict (around line 228) with:
```python
"transition_after": transition_after, "transition_used": chosen[0] == "transition",
"transition_error": transition_error,
```

Update the `song` dict (around line 246) and the `candidates` comprehension (line 250) with `transition_used` / `transition_after` / `transition_error`.

- [ ] **Step 2: Run the existing test suite**

Run: `tools/vocal_repair/.venv/Scripts/python.exe -m pytest tools/vocal_repair/tests/ -v`
Expected: PASS (existing 21 + 6 new)

- [ ] **Step 3: Commit**

```bash
git add tools/vocal_repair/quality_rebuild_agent.py
git commit -m "feat: integrate transition repair into quality rebuild agent"
```

---

### Task 6: Integrate into FastAPI queue.py + config

**Files:**
- Modify: `services/api/app/services/queue.py`
- Modify: `services/api/app/config.py`

- [ ] **Step 1: Add config setting**

In `services/api/app/config.py` near line 28:

```python
    vocal_transition_repair_enabled: bool = True
```

- [ ] **Step 2: Add transition repair hook to queue.py**

In `services/api/app/services/queue.py` after `humanize_generated`, add:

```python
TRANSITION_SCRIPT = PROJECT_ROOT / "tools" / "vocal_repair" / "transition_repair.py"


async def transition_repair_generated(output_path: Path) -> Path:
    """Post-process a generated song: demucs -> frame repair -> noise floor ->
    transition repair -> de_buzz EQ -> remix.

    Falls back to the raw output on any failure so generation never breaks.
    """
    if not TOOLS_VENV_PYTHON.exists():
        logger.warning("vocal_repair venv missing at %s; skipping transition repair", TOOLS_VENV_PYTHON)
        return output_path
    work_dir = PROJECT_ROOT / "outputs" / "humanize_work" / output_path.stem
    cmd = [
        str(TOOLS_VENV_PYTHON),
        str(TRANSITION_SCRIPT),
        str(output_path),
        str(output_path),  # input == output (in-place repair, like humanize)
        "--max-gap-ms", "300",
        "--work-dir", str(work_dir),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=1800)
        if proc.returncode != 0:
            logger.error("transition repair failed: %s", stderr.decode()[:2000])
            return output_path
        logger.info("transition repair applied: %s", stdout.decode().strip()[-200:])
    except Exception as exc:  # noqa: BLE001 - repair must never break generation
        logger.warning("transition repair error (falling back to raw output): %s", exc)
        return output_path
    return output_path
```

And in `_process_job` (around line 113), wrap the humanize call:

```python
if settings.vocal_transition_repair_enabled and not job.instrumental:
    output_path = await transition_repair_generated(Path(output_path))
```

- [ ] **Step 3: Verify the service command end-to-end (optional, GPU-free)**

Run the transition repair CLI in full mode on the mixed 02 file (no service startup needed):
```bash
tools/vocal_repair/.venv/Scripts/python.exe tools/vocal_repair/transition_repair.py \
  outputs/ab_compare/02_HeartMuLa_60s_moonlit.wav \
  outputs/transition_work/mix_transition_repaired_02.wav \
  --work-dir outputs/transition_work/queue_work
```
Expected: prints "full pipeline: repaired N events -> ..." (N >= 1), no crash. The `queue_work` dir is cleaned up after (keep_work=False default in `_process_job`).

- [ ] **Step 4: Commit**

```bash
git add services/api/app/config.py services/api/app/services/queue.py
git commit -m "feat: wire transition repair into FastAPI post-generation hook"
```

---

### Task 7: Gate regression + final verification

**Files:**
- none (verification)

- [ ] **Step 1: Run full test suite**

Run: `tools/vocal_repair/.venv/Scripts/python.exe -m pytest tools/vocal_repair/tests/ -v`
Expected: all pass (21 existing + 6 new = 27)

- [ ] **Step 2: Run the quality rebuild agent on the HeartMuLa 60s candidate**

Run (in the vocal_repair venv):
```bash
tools/vocal_repair/.venv/Scripts/python.exe tools/vocal_repair/quality_rebuild_agent.py \
  --input-dir outputs/ab_compare --review-dir outputs/transition_work/review_gate --work-dir outputs/transition_work/gate_work
```
Expected: manifest `outputs/transition_work/review_gate/quality_manifest.json` shows the HeartMuLa 60s song either PASSED (transition_used=true or humanize_used=true) or was rejected with a reason. Confirm artifact/gate/noise metrics do not regress vs the pre-transition baseline (compare `quality_value` with the same song before this change).

- [ ] **Step 3: Verify no orphan processes / temp cleanup**

Check that `outputs/transition_work/gate_work` was removed (keep_work=False default), and no lingering python/ffmpeg processes (tasklist | grep python).

- [ ] **Step 4: Commit any tweaks**

```bash
git add -A
git commit -m "chore: transition repair gate regression verification"
```

---

## Self-Review Notes

- **Spec coverage:** detector (Task 1), glide builder (Task 2), WORLD blend (Task 2), CLI raw + full modes (Task 3), pipeline integration (Tasks 5-6), tests (Tasks 1-2, 7), A/B gate (Task 4), error handling (Task 2 try/except + Task 6 fallback), long-gap flagging (Task 2 report + Task 4).
- **Environment:** pyworld API verified (`cheaptrick` lowercase, `synthesize` signature `synthesize(f0, sp, ap, sr, frame_period)`).
- **CLI contract:** Task 3 defines `main()` with raw mode `transition_repair.py <vocals.wav> <out.wav> [--report]` and full mode `transition_repair.py <song.wav> <out.wav> --work-dir ...` (demucs + frame repair + noise floor + transition repair + remix). Task 6 queue hook uses full mode with input==output for in-place repair, matching `humanize_vocal.py`'s `--output` contract.
- **Signatures consistent:** `extract_f0(x, sr)`, `detect_transition_events(f0, sr, x=x)`, `build_glide_f0(f_a, f_b, t0, t1, sr)`, `repair_transitions(x, sr, max_gap_s)`, `analyze_world(x, sr)`, `synthesize_window(x, sr, world, ev, glide)`.
