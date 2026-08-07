"""Surgical pitch-transition ("转音台阶") repair for AI vocal stems.

Detects three defect classes at note transitions (pitch step without glide,
true-silence gap, unpitched screech segment) and resynthesizes only the
detected windows with a WORLD portamento glide, leaving everything else
bit-exact. See docs/superpowers/specs/2026-08-07-transition-repair-design.md.

Usage:
    python transition_repair.py <vocals.wav> <out.wav> [--max-gap-ms N] [--report report.json]
    python transition_repair.py <song.wav> <out.wav> --work-dir <dir>  # full pipeline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

HOP = 256
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


def st_of(hz: float | np.ndarray) -> float | np.ndarray:
    return 12 * np.log2(np.maximum(hz, 1.0) / 440.0) + 69.0


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
                # Sample the bridging energy on the window MIDDLE only: the
                # inter-note window's edges are the notes' fades/onsets, which
                # carry energy even when the middle is true silence. Measuring
                # the whole window reads the note edges and misclassifies a
                # long true-silence gap as energy-bridged (the real HeartMuLa
                # 39.2s bug: ~1.7s of -66..-85dB silence flanked by note A's
                # tail and note B's onset — which starts ~370ms INSIDE the F0
                # gap window at -15.8dB — was typed screech and "repaired"
                # with a 2.3s synthesized glide). Exclude a ~100ms margin per
                # side — adaptive (30% of the window) so short windows keep a
                # proportionally smaller margin and the middle never collapses.
                margin = min(int(0.100 * sr), int((i1 - i0) * 0.3))
                mid_i0 = i0 + margin
                mid_i1 = i1 - margin
                if mid_i1 <= mid_i0:
                    # margins would swallow the window entirely; the
                    # min-window adjacent-notes case has no silence in the
                    # middle anyway, so fall back to the whole window
                    mid_i0, mid_i1 = i0, i1
                rms = []
                for m in range(mid_i0, mid_i1, w // 2):
                    m2 = min(m + w, mid_i1)  # never overrun the middle region
                    if m2 <= m:
                        continue
                    rms.append(20 * np.log10(np.sqrt(np.mean(x[m:m2] ** 2)) + 1e-9))
                rms_db = np.asarray(rms)
                # silence_gap when the MIDDLE is dominated by true silence
                # (median, not max: a real dropout's sung onset can extend
                # hundreds of ms into the F0 gap window, so 'any loud window'
                # would misread it as bridged). The bridged branch keeps the
                # any-window max test: noise that bridges a gap is a screech
                # defect even where it has quiet patches.
                if rms_db.size and np.median(rms_db) < silence_db:
                    etype = "silence_gap"
                elif rms_db.size and rms_db.max() >= screech_rms_db:
                    bridged = True  # energy bridges the gap (noise segment)
                    mid0, mid1 = a["end"], b["start"]
                    if mid1 > mid0 and (mid1 - mid0) * frame_s >= screech_min_s:
                        region = f0[mid0:mid1]
                        if np.any(region == 0) or _erratic(region):
                            etype = "screech"
        # The glide budget gates clean steps only: a long unvoiced-but-not-
        # silent step gap is dropped as unrepairable. A true-silence gap
        # longer than the budget must still surface so repair can flag it
        # long_gap_skipped instead of bridging it (spec: longer gaps are
        # flagged in the report, not bridged). A noise-bridged gap is a
        # screech defect regardless of length.
        if gap_s > glide_max_s and not bridged and etype != "silence_gap":
            j += 1
            continue
        events.append({
            "type": etype, "t0": float(t0), "t1": float(t1),
            "f_a": float(a["f0"]), "f_b": float(b["f0"]),
            "jump_st": float(jump), "flags": [],
        })
        j += 1
    return events


def build_glide_f0(f_a: float, f_b: float, t0: float, t1: float,
                   sr: int, hold_frac: float = 0.3) -> np.ndarray:
    """Exponential glide F0 over [t0, t1] with natural shape.

    Holds f_a for the first hold_frac of the window, then moves from f_a to
    f_b on a smoothstep (3g^2 - 2g^3) in semitone space. The smoothstep's
    output stays within [0, 1] of its argument, so the trajectory is a
    monotonically curved portamento that stays between st_a and st_b and
    converges exactly onto f_b; the test contract requires strict monotonic
    st() (no true overshoot, which would force a downward settle).
    Returns per-frame F0 at HOP spacing, indexed from t0.
    """
    n_frames = max(1, int(round((t1 - t0) * sr / HOP)))
    k = np.arange(n_frames) / max(n_frames - 1, 1)
    st_a, st_b = st_of(f_a), st_of(f_b)
    # hold + smoothstep rise (bounded in [st_a, st_b] by construction)
    g = np.clip((k - hold_frac) / (1 - hold_frac), 0, 1)
    st_shape = st_a + (st_b - st_a) * (3.0 * g ** 2 - 2.0 * g ** 3)
    f0 = 440.0 * 2 ** ((st_shape - 69.0) / 12.0)
    # edges converge exactly to the notes
    f0[0] = f_a
    f0[-1] = f_b
    return f0


def analyze_world(x: np.ndarray, sr: int) -> dict:
    """Full-stem WORLD analysis: f0, spectral envelope, aperiodicity."""
    import pyworld as pw
    frame_period = HOP * 1000.0 / sr
    f0, tpos = pw.harvest(x.astype(np.float64), sr, frame_period=frame_period,
                          f0_floor=F0_MIN, f0_ceil=F0_MAX)
    # pyworld 0.3.5 signatures (verified in the venv): cheaptrick/d4c take
    # (x, f0, temporal_positions, fs, ...) — harvest's tpos is required;
    # there is no frame_period kwarg form, it raises TypeError.
    sp = pw.cheaptrick(x.astype(np.float64), f0, tpos, sr)
    ap = pw.d4c(x.astype(np.float64), f0, tpos, sr)
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
        # tile the window's own frame if the analysis ran short (never crashes)
        sp0 = world["sp"][min(fr0, len(world["sp"]) - 1)]
        spw = world["sp"][fr0:fr0 + n_frames] if fr0 + n_frames <= len(world["sp"]) \
            else np.tile(sp0, (n_frames, 1))
    # aperiodicity: screech blends toward surrounding notes
    if ev["type"] == "screech":
        ap_a = world["ap"][fr0 - 1] if fr0 - 1 >= 0 else world["ap"][0]
        ap_b = world["ap"][fr1] if fr1 < len(world["ap"]) else world["ap"][-1]
        apw = np.stack([
            ap_a + (ap_b - ap_a) * (k / max(n_frames - 1, 1))
            for k in range(n_frames)
        ])
    else:
        ap0 = world["ap"][min(fr0, len(world["ap"]) - 1)]
        apw = world["ap"][fr0:fr0 + n_frames] if fr0 + n_frames <= len(world["ap"]) \
            else np.tile(ap0, (n_frames, 1))
    y = pw.synthesize(f0w, spw, apw, sr, frame_period)
    # align to the window length: pyworld returns n_frames*HOP samples, and
    # rounding n_frames up (e.g. the 100ms min window at 48kHz gives 19
    # frames = 4864 > 4800) can overshoot the window by up to HOP-1 samples.
    # The excess sits inside the right crossfade tail, so truncating is safe.
    target = i1 - i0
    y = np.pad(y, (0, max(0, target - len(y))))
    if len(y) > target:
        y = y[:target]
    return y


def _crossfade(a: np.ndarray, b: np.ndarray, sr: int, fade_s: float = 0.030) -> np.ndarray:
    """Blend a (original) and b (repaired window): left edge fades a->b,
    right edge fades b->a so the output ends on the original for continuity,
    and the middle is b — the repair must reach the window interior."""
    nf = min(int(fade_s * sr), len(a), len(b))
    if nf <= 0:
        return b
    out = b.copy()
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
    # The detector's glide budget must match the caller's max_gap_s, or gaps
    # longer than the hard-coded 0.30s would be dropped before repair sees
    # them and the long_gap_skipped flag would be dead with defaults.
    events = detect_transition_events(f0, sr, x=x, glide_max_s=max_gap_s)
    report = {"events": events, "repaired": []}
    if not events:
        return x.copy(), report
    world = analyze_world(x, sr)
    out = x.copy()
    for ev in events:
        # The max_gap_s cap gates clean step/silence gaps only; screech
        # windows are energy-bridged (noise, not silence) so they are always
        # repaired regardless of length (see detect_transition_events).
        if ev["type"] != "screech" and ev["t1"] - ev["t0"] > max_gap_s:
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


def _apply_stereo(y: np.ndarray, out_mono: np.ndarray, report: dict,
                  sr: int) -> np.ndarray:
    """Blend the repaired mono into the original channels, window-only.

    Keeps each channel's original samples everywhere except the actually
    repaired event windows (report entries with skipped=False), where the
    repaired mono replaces both channels. This preserves the design's
    construction guarantee (bit-exact outside windows) for real stereo
    stems — mono-mixing the whole file would rewrite every sample.
    """
    if y.ndim == 1:
        return np.clip(out_mono, -1, 1)
    in_win = np.zeros(len(y), dtype=bool)
    for e in report["repaired"]:
        if not e["skipped"]:
            i0 = int(e["t0"] * sr)
            i1 = min(int(e["t1"] * sr), len(y))
            if i1 > i0:
                in_win[i0:i1] = True
    final = y.astype(np.float64)
    final[in_win, 0] = out_mono[in_win]
    final[in_win, 1] = out_mono[in_win]
    return np.clip(final, -1, 1)


def _repair_wav(input_wav: Path, output: Path, max_gap_s: float,
                report_path: Path | None) -> dict:
    """Repair a raw vocal stem file, write PCM_16 WAV (+ optional JSON report)."""
    y, sr = sf.read(input_wav, always_2d=True)
    x = y.mean(axis=1).astype(np.float64) if y.ndim > 1 else y.astype(np.float64)
    out, report = repair_transitions(x, sr, max_gap_s=max_gap_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, _apply_stereo(y, out, report, sr), sr, subtype="PCM_16")
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair pitch-transition defects on a vocal stem")
    parser.add_argument("input", type=Path,
                        help="vocal stem (raw mode) or song WAV (full mode)")
    parser.add_argument("output", type=Path, help="output WAV path")
    parser.add_argument("--max-gap-ms", type=float, default=300.0,
                        help="longest clean gap (ms) to bridge; longer gaps are "
                             "flagged long_gap_skipped, not repaired")
    parser.add_argument("--report", type=Path, default=None,
                        help="write the event/repair JSON report here")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="full mode: demucs/frame/noise/transition work dir")
    args = parser.parse_args()

    if args.work_dir is not None:
        # full mode: demucs separate -> frame repair -> noise floor ->
        # transition repair -> remix (mirrors the FastAPI queue hook, Task 6).
        # stem_pipeline is imported lazily so raw mode never needs demucs/ffmpeg.
        from stem_pipeline import remix, separate_with_demucs
        song_work = args.work_dir / args.input.stem
        vocals, instrumental = separate_with_demucs(args.input, song_work)
        from frame_repair import repair_vocals
        from noise_floor_reducer import reduce_noise_floor
        reduced = song_work / "vocals_noise_reduced.wav"
        reduce_noise_floor(vocals, reduced)
        repaired = song_work / "vocals_frame_repaired.wav"
        repair_vocals(reduced, repaired)
        y2, sr2 = sf.read(repaired, always_2d=True)
        x2 = y2.mean(axis=1).astype(np.float64) if y2.ndim > 1 else y2.astype(np.float64)
        out2, report = repair_transitions(x2, sr2, max_gap_s=args.max_gap_ms / 1000.0)
        staged = song_work / "vocals_transition_repaired.wav"
        sf.write(staged, _apply_stereo(y2, out2, report, sr2), sr2, subtype="PCM_16")
        remix(instrumental, staged, args.output)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"full pipeline: repaired {len(report['repaired'])} events -> {args.output}")
        return 0

    # raw mode: single vocal stem
    report = _repair_wav(args.input, args.output, args.max_gap_ms / 1000.0,
                         args.report)
    print(f"repaired {len(report['repaired'])} events -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
