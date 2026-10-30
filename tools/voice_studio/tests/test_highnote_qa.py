import numpy as np
import pytest

from tools.voice_studio import highnote_qa as qa


def _vibrato_tone(cents=30.0, rate=44100, seconds=1.6, f0=880.0):
    t = np.arange(int(rate * seconds)) / rate
    deviation = f0 * (2 ** (cents / 1200.0) - 1.0)
    phase = 2 * np.pi * f0 * t + (deviation / 5.5) * np.sin(2 * np.pi * 5.5 * t)
    return 0.3 * np.sin(phase) + 0.1 * np.sin(2 * phase)


def _glide(start=600.0, end=1000.0, rate=44100, seconds=1.6, octave=1.0):
    t = np.arange(int(rate * seconds)) / rate
    slope = (end - start) / seconds
    phase = 2 * np.pi * octave * (start * t + 0.5 * slope * t * t)
    return 0.3 * np.sin(phase)


def test_synthetic_self_test_passes():
    result = qa.self_test()
    assert result["status"] == "passed"
    assert result["vibrato_depth_cents"] > 15
    assert result["flat_ratio_constant"] == 1.0


def test_band_limited_24k_is_flagged():
    tone = _vibrato_tone()
    result = qa.analyze(tone, 44100)
    np.testing.assert_allclose(result["rate"], 44100)
    limited = qa._resample(tone, 44100, 24000)
    flagged = qa.analyze(limited, 24000)
    assert "band_limited_24k" in flagged["flags"]
    assert "band_limited_24k" not in result["flags"]


def test_flat_long_note_is_flagged_only_without_vibrato():
    with_vibrato = qa.analyze(_vibrato_tone(30), 44100)
    flat = qa.analyze(_vibrato_tone(0), 44100)
    assert "flat_vibrato_high_note" not in with_vibrato["flags"]
    assert "flat_vibrato_high_note" in flat["flags"]


def test_harmonic_ratio_separates_inharmonic_noise():
    rate = 24000
    t = np.arange(rate) / rate
    f0 = 937.0
    clean = (0.3 * np.sin(2 * np.pi * f0 * t)
             + 0.1 * np.sin(4 * np.pi * f0 * t)
             + 0.05 * np.sin(6 * np.pi * f0 * t))
    noisy = clean + 0.05 * np.random.default_rng(0).normal(size=len(clean))
    contour = np.full(int(rate / 93.75), f0)
    clean_ratio = qa._harmonic_ratio(clean, rate, contour, 93.75)
    noisy_ratio = qa._harmonic_ratio(noisy, rate, contour, 93.75)
    assert clean_ratio is not None and noisy_ratio is not None
    assert clean_ratio > noisy_ratio + 20.0


def test_octave_error_is_visible_in_the_comparison():
    candidate = _glide()
    source = _glide(octave=0.5)
    result = qa.analyze(candidate, 44100, source=source, source_rate=44100)
    assert result["comparison"]["octave_error_ratio"] > 0.5
    assert "pitch_tracking_mismatch" in result["flags"]


def test_identical_source_has_no_tracking_flag():
    tone = _glide()
    result = qa.analyze(tone, 44100, source=tone, source_rate=44100)
    assert result["comparison"]["f0_contour_correlation"] > 0.99
    assert "pitch_tracking_mismatch" not in result["flags"]


def test_silence_and_nonfinite_are_rejected():
    with pytest.raises(ValueError, match="No voiced frames"):
        qa.analyze(np.zeros(44100), 44100)
    bad = _vibrato_tone()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        qa.analyze(bad, 44100)
