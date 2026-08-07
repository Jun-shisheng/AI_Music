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


def test_silence_gap_classified_with_audio(tmp_path):
    """True-silence gap between notes is classified silence_gap, not
    screech: the 25ms RMS window must not overrun into note B's onset."""
    gap_s = 0.2
    sig, sr = make_audio_with_step(tmp_path, gap_s=gap_s)
    f0 = extract_f0(sig, sr)
    events = detect_transition_events(f0, sr, x=sig)
    assert len(events) == 1
    assert events[0]["type"] == "silence_gap"


def test_screech_classified_with_noise_between_notes(tmp_path):
    """Unpitched noise burst between two stable notes is classified
    screech: harvest locks F0 onto the noise erratically (or drops out),
    so the energy-bridged region is a screech defect."""
    sig, sr = make_audio_with_step(tmp_path, gap_s=0.25)
    gap = int(0.25 * SR)
    i0 = int(0.4 * SR)  # note A duration, noise goes into the silent gap
    rng = np.random.default_rng(7)
    from scipy import signal as sig_module
    noise = rng.normal(0, 1, gap)
    sos = sig_module.butter(4, [2000 / (SR / 2), 8000 / (SR / 2)],
                            btype="bandpass", output="sos")
    band_noise = sig_module.sosfilt(sos, noise) * 0.15
    sig[i0:i0 + gap] += band_noise
    f0 = extract_f0(sig, sr)
    events = detect_transition_events(f0, sr, x=sig)
    assert len(events) == 1
    assert events[0]["type"] == "screech"


def test_adjacent_zero_gap_notes_detected_from_f0():
    """Two fully-voiced adjacent notes (zero-length gap) must split cleanly:
    the running median is computed over the current segment only, so the
    6-st step produces exactly one step event with the min window enforced."""
    f0 = np.concatenate([np.full(60, 220.0), np.full(60, 311.127)])
    events = detect_transition_events(f0, SR)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "step"
    assert e["t1"] - e["t0"] >= 0.08
    assert abs(ST(e["f_a"]) - ST(220.0)) < 1.5
    assert abs(ST(e["f_b"]) - ST(311.127)) < 1.5


def test_all_voiced_erratic_bridge_classified_as_screech():
    """An all-voiced, harvest-locked-noise bridge between two stable notes
    must be classified screech via the erratic-F0 branch: rapid frame-to-frame
    pitch jumps with zero unvoiced frames (so np.any(region == 0) is False and
    _erratic() must decide). Exercises the ndarray path of st_of()."""
    rng = np.random.default_rng(42)
    walk = np.empty(20)
    walk[0] = 400.0
    for k in range(1, 20):
        walk[k] = min(700.0, max(150.0, walk[k - 1] + rng.uniform(-180, 180)))
    f0 = np.concatenate([np.full(40, 220.0), walk, np.full(40, 311.127)])
    assert np.all(f0 > 0)  # the bridge must be fully voiced
    n = int(len(f0) * FRAME_S * SR)
    x = rng.normal(0, 1, n)  # any nonzero-energy audio passes the RMS bridge check
    events = detect_transition_events(f0, SR, x=x)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "screech"
    assert e["t1"] - e["t0"] >= len(walk) * FRAME_S - 0.02


def test_single_off_pitch_boundary_frame_tolerated():
    """One off-pitch frame at the note boundary must not fragment the
    segmentation: only >= 2 consecutive deviant frames start a new note."""
    f0 = np.concatenate([np.full(60, 220.0), np.array([287.5]),
                         np.full(59, 311.127)])
    events = detect_transition_events(f0, SR)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "step"
    assert e["t1"] - e["t0"] >= 0.08
    assert abs(ST(e["f_a"]) - ST(220.0)) < 1.5
    assert abs(ST(e["f_b"]) - ST(311.127)) < 1.5
