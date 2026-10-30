import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanize_vocal import HUMANIZE_VARIANTS, humanize_stem  # noqa: E402


def test_humanize_variants_defined():
    assert "de_buzz" in HUMANIZE_VARIANTS
    chain = HUMANIZE_VARIANTS["de_buzz"]
    assert "deesser" in chain
    assert "highshelf" in chain
    assert "equalizer=f=280" in chain


def test_humanize_stem_produces_audio(tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 44100
    t = np.arange(int(sr * 1.0)) / sr
    sig = 0.5 * np.sin(2 * np.pi * 220 * t)
    src = tmp_path / "vocal.wav"
    sf.write(src, np.stack([sig, sig], axis=1), sr, subtype="PCM_16")

    out = tmp_path / "humanized.wav"
    humanize_stem(src, out, variant="de_buzz")
    assert out.exists()
    y, _ = sf.read(out, always_2d=True)
    assert len(y) > 0
    assert abs(np.max(np.abs(y))) <= 1.0  # limiter keeps it bounded
    assert not (tmp_path / "humanized_frame_repaired.wav").exists()  # temp removed


def test_humanize_stem_keeps_harsh_events_low_after_eq(tmp_path):
    """Regression: the de_buzz EQ (7 kHz air shelf) re-introduces harsh HF
    noise when applied after frame repair. Frame repair must run last so
    defect regions are suppressed after the EQ stage."""
    import numpy as np
    import soundfile as sf
    from scipy import signal as sig_module

    from artifact_segments import detect_events

    sr = 44100
    dur = 2.0
    t = np.arange(int(sr * dur)) / sr
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)
           + 0.2 * np.sin(2 * np.pi * 440 * t)
           + 0.1 * np.sin(2 * np.pi * 660 * t))
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 1, len(t))
    sos = sig_module.butter(4, [4000 / (sr / 2), 10000 / (sr / 2)], btype="bandpass", output="sos")
    band_noise = sig_module.sosfilt(sos, noise)
    mask = np.zeros(len(t))
    mask[int(0.7 * sr):int(0.9 * sr)] = 1.0
    sig = sig + band_noise * mask
    src = tmp_path / "noisy_vocal.wav"
    sf.write(src, np.stack([sig, sig], axis=1), sr, subtype="PCM_16")

    out = tmp_path / "humanized.wav"
    humanize_stem(src, out, variant="de_buzz")

    # the repair must leave no audible harsh frames in the noisy region
    events = detect_events(out, max_events_per_file=1000)
    harsh = [e for e in events if e["type"] == "harsh_machine_noise" and float(e["severity"]) >= 0.55]
    in_noise_zone = [e for e in harsh if 0.65 <= float(e["time_seconds"]) <= 0.95]
    assert len(in_noise_zone) == 0, f"harsh frames survived in EQ'd region: {in_noise_zone}"
