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


_HARM_COEFS = [0.5, 0.25, 0.15, 0.08]  # fundamental + 2nd..4th partial
_FADE_S = 0.02  # raised-cosine edge fade, per note


def _tone(f0, dur_s, sr):
    """Synthetic 'vocal' tone: fundamental + harmonics + edge fades.

    pyworld harvest 0.3.5 behaves pathologically on ultra-clean synthetic
    signals: it returns all-unvoiced for a pure sine or sparse harmonic
    stack, and emits spurious 1-2 frame F0 notes / drops out at note edges
    without a fade. The 4-partial stack plus short edge fades (which real
    sung notes have anyway) make it lock F0 cleanly.
    """
    t = np.arange(int(dur_s * sr)) / sr
    sig = np.zeros_like(t)
    for k, c in enumerate(_HARM_COEFS, start=1):
        sig += c * np.sin(2 * np.pi * k * f0 * t)
    nf = int(_FADE_S * sr)
    sig[:nf] *= np.linspace(0, 1, nf)
    sig[-nf:] *= np.linspace(1, 0, nf)
    return sig


def make_audio_with_step(tmp_path, gap_s=0.2, jump_st=6.0, f0_a=220.0):
    """Two stable notes with a hard pitch step and optional silent gap."""
    note_dur = 0.4
    sr = SR
    gap = int(gap_s * sr)
    f0_b = f0_a * (2 ** (jump_st / 12))
    sig = np.concatenate([
        _tone(f0_a, note_dur, sr),
        np.zeros(gap),
        _tone(f0_b, note_dur, sr),
    ])
    return sig.astype(np.float64), sr


def test_detects_pitch_step_with_silent_gap(tmp_path):
    gap_s = 0.2
    sig, sr = make_audio_with_step(tmp_path, gap_s=gap_s)
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
    """Notes nearly touching (20ms micro-gap) still yield a step + min window.

    gap_s=0.0 is untestable through pyworld: a phase-continuous boundary
    makes harvest 0.3.5 mis-track or drop out entirely. 20ms is the smallest
    gap it tracks cleanly; it is below the detector's 25ms silence window and
    below the 100ms minimum event window, so the min-window path (intended
    for zero-length gaps) is still exercised.
    """
    sig, sr = make_audio_with_step(tmp_path, gap_s=0.02)
    f0 = extract_f0(sig, sr)
    events = detect_transition_events(f0, sr)
    steps = [e for e in events if e["type"] == "step"]
    assert len(steps) >= 1
    # minimum window enforced
    assert steps[0]["t1"] - steps[0]["t0"] >= 0.08


def test_no_events_on_steady_sine(tmp_path):
    sig = _tone(220.0, 1.0, SR)
    f0 = extract_f0(sig, SR)
    events = detect_transition_events(f0, SR)
    assert len(events) == 0
