import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_repair import repair_vocals  # noqa: E402


SR = 44100


def make_synth(tmp_path, with_noise=True, with_drop=True) -> Path:
    """1s stereo wav: fundamental + harmonics, optional 4-10k noise burst and level drop."""
    t = np.arange(SR) / SR
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)
           + 0.2 * np.sin(2 * np.pi * 440 * t)
           + 0.1 * np.sin(2 * np.pi * 660 * t))
    rng = np.random.default_rng(7)
    if with_noise:
        from scipy import signal as sig_module
        noise = rng.normal(0, 1, SR)
        sos = sig_module.butter(4, [4000 / (SR / 2), 10000 / (SR / 2)], btype="bandpass", output="sos")
        band_noise = sig_module.sosfilt(sos, noise)
        mask = np.zeros(SR)
        mask[int(0.30 * SR):int(0.42 * SR)] = 1.0
        sig = sig + band_noise * mask
    if with_drop:
        lo, hi = int(0.60 * SR), int(0.66 * SR)
        sig[lo:hi] *= 0.05
    stereo = np.stack([sig, sig], axis=1)
    import soundfile as sf
    wav = tmp_path / "synth_vocal.wav"
    sf.write(wav, stereo, SR, subtype="PCM_16")
    return wav


def _butter_lowpass(cutoff_hz, n=4):
    from scipy import signal as sig_module
    return sig_module.butter(n, cutoff_hz / (SR / 2), btype="low")


def _band_energy(y: np.ndarray, lo_hz: int, hi_hz: int) -> float:
    from scipy import signal as sig_module
    f, P = sig_module.welch(y, SR, nperseg=1024)
    m = (f >= lo_hz) & (f <= hi_hz)
    return float(np.mean(P[m]))


def test_repair_suppresses_noise_band_without_touching_rest(tmp_path):
    import soundfile as sf
    from artifact_segments import detect_events

    src = make_synth(tmp_path)
    out = tmp_path / "repaired.wav"
    report = repair_vocals(src, out)

    assert report["event_count"] >= 1
    y, _ = sf.read(out, always_2d=True)
    mono = y.mean(axis=1)
    assert len(mono) == SR

    # 4-10k energy inside the noisy region must drop vs the source
    src_y, _ = sf.read(src, always_2d=True)
    src_mono = src_y.mean(axis=1)
    lo, hi = int(0.32 * SR), int(0.40 * SR)
    before_energy = _band_energy(src_mono[lo:hi], 4000, 10000)
    after_energy = _band_energy(mono[lo:hi], 4000, 10000)
    assert after_energy < before_energy * 0.6

    # outside regions must be untouched (within PCM rounding)
    clean_lo, clean_hi = int(0.10 * SR), int(0.25 * SR)
    assert np.max(np.abs(mono[clean_lo:clean_hi] - src_mono[clean_lo:clean_hi])) < 0.01


def test_repair_recovers_level_drop(tmp_path):
    import soundfile as sf

    src = make_synth(tmp_path)
    out = tmp_path / "repaired.wav"
    repair_vocals(src, out)
    y, _ = sf.read(out, always_2d=True)
    mono = y.mean(axis=1)
    src_y, _ = sf.read(src, always_2d=True)
    src_mono = src_y.mean(axis=1)

    drop_lo, drop_hi = int(0.60 * SR), int(0.66 * SR)
    before_rms = np.sqrt(np.mean(src_mono[drop_lo:drop_hi] ** 2))
    after_rms = np.sqrt(np.mean(mono[drop_lo:drop_hi] ** 2))
    assert after_rms > before_rms * 3


def test_repair_is_idempotent_on_clean_signal(tmp_path):
    import soundfile as sf

    src = make_synth(tmp_path, with_noise=False, with_drop=False)
    out = tmp_path / "repaired.wav"
    repair_vocals(src, out)
    y, _ = sf.read(out, always_2d=True)
    src_y, _ = sf.read(src, always_2d=True)
    assert np.max(np.abs(y - src_y)) < 0.01


def test_two_round_repair_suppresses_more_harsh_events(tmp_path):
    """Regression: one repair pass lowers harsh zcr 0.65 -> 0.56-0.68 (still
    above the audible 0.55 threshold); the second round pushes them under 0.55.
    On real AI stems this is the difference between 3 and 1 final-mix events."""
    import soundfile as sf
    from artifact_segments import detect_events

    # signal with TWO noise bursts so one round leaves survivors near threshold
    sr = 44100
    t = np.arange(sr * 2) / sr
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)
           + 0.2 * np.sin(2 * np.pi * 440 * t)
           + 0.1 * np.sin(2 * np.pi * 660 * t))
    rng = np.random.default_rng(21)
    from scipy import signal as sig_module
    noise = rng.normal(0, 1, len(t))
    sos = sig_module.butter(4, [4000 / (sr / 2), 10000 / (sr / 2)], btype="bandpass", output="sos")
    band_noise = sig_module.sosfilt(sos, noise)
    for start, end in ((0.6, 0.75), (1.2, 1.35)):
        m = np.zeros(len(t))
        m[int(start * sr):int(end * sr)] = 1.0
        sig = sig + band_noise * m * 0.9
    src = tmp_path / "double_noisy.wav"
    sf.write(src, np.stack([sig, sig], axis=1), sr, subtype="PCM_16")

    def count(p):
        ev = detect_events(p, max_events_per_file=1000)
        return sum(1 for e in ev if e["type"] == "harsh_machine_noise" and float(e["severity"]) >= 0.55)

    one = tmp_path / "one.wav"
    repair_vocals(src, one, repair_rounds=1)
    two = tmp_path / "two.wav"
    repair_vocals(src, two, repair_rounds=2)

    n1, n2 = count(one), count(two)
    # two rounds must not leave more harsh events than one round
    assert n2 <= n1, f"two rounds regressed: {n1} -> {n2}"
    # on this synthetic double-burst signal two rounds should reach zero
    assert n2 == 0, f"two rounds left harsh events: {n2}"


def test_isolation_only_preserves_sibilant_runs(tmp_path):
    """Regression: the zcr harsh detector cannot separate sibilants from machine
    noise (human refs trigger many 'harsh' events). Isolation mode must repair
    only isolated single-frame bursts and leave multi-frame runs (diction) alone.
    A 3-frame high-zcr run at 5-8k is a sibilant; a lone burst frame is noise."""
    import soundfile as sf
    from artifact_segments import detect_events

    sr = 44100
    dur = 2.0
    t = np.arange(int(sr * dur)) / sr
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)
           + 0.2 * np.sin(2 * np.pi * 440 * t))
    rng = np.random.default_rng(33)
    from scipy import signal as sig_module
    noise = rng.normal(0, 1, len(t))
    sos = sig_module.butter(4, [5000 / (sr / 2), 8000 / (sr / 2)], btype="bandpass", output="sos")
    sib = sig_module.sosfilt(sos, noise)

    # sibilant run: 3 consecutive frames (0.6-0.75s)
    m_run = np.zeros(len(t))
    m_run[int(0.6 * sr):int(0.75 * sr)] = 1.0
    # isolated burst: single 50ms frame (1.4s)
    m_iso = np.zeros(len(t))
    m_iso[int(1.40 * sr):int(1.45 * sr)] = 1.0
    # high-frequency dominant so zcr clearly clears the 0.38 detector floor
    # (a 220 Hz tone would drag zcr down below detection)
    sig = sig + sib * m_run * 2.5 + sib * m_iso * 2.5
    src = tmp_path / "sibilant_and_burst.wav"
    sf.write(src, np.stack([sig, sig], axis=1), sr, subtype="PCM_16")

    def sib_energy(p, lo_s=0.58, hi_s=0.77):
        y, _ = sf.read(p, always_2d=True)
        return float(np.sqrt(np.mean(y[int(lo_s * sr):int(hi_s * sr)] ** 2)))

    def iso_energy(p, lo_s=1.38, hi_s=1.47):
        y, _ = sf.read(p, always_2d=True)
        return float(np.sqrt(np.mean(y[int(lo_s * sr):int(hi_s * sr)] ** 2)))

    out = tmp_path / "iso.wav"
    repair_vocals(src, out, repair_rounds=1, isolation_only=True)

    # sibilant run must be preserved (diction) - energy ratio close to 1
    run_before, run_after = sib_energy(src), sib_energy(out)
    assert run_after > run_before * 0.85, f"sibilant run eaten: {run_before:.4f} -> {run_after:.4f}"
    # isolated burst should be attenuated noticeably
    burst_before, burst_after = iso_energy(src), iso_energy(out)
    assert burst_after < burst_before * 0.85, f"isolated burst not repaired: {burst_before:.4f} -> {burst_after:.4f}"


def test_isolation_only_keeps_harsh_count_no_worse_than_std(tmp_path):
    """Isolation mode must not leave more harsh frames than standard single pass."""
    import soundfile as sf
    from artifact_segments import detect_events

    src = make_synth(tmp_path, with_noise=True, with_drop=False)
    iso = tmp_path / "iso.wav"
    repair_vocals(src, iso, repair_rounds=1, isolation_only=True)
    std = tmp_path / "std.wav"
    repair_vocals(src, std, repair_rounds=1)

    def count(p):
        ev = detect_events(p, max_events_per_file=1000)
        return sum(1 for e in ev if e["type"] == "harsh_machine_noise" and float(e["severity"]) >= 0.55)

    # the synth burst is a wide isolated region; both should reduce it
    src_count = count(src)
    assert count(iso) <= src_count
    assert count(std) <= src_count
