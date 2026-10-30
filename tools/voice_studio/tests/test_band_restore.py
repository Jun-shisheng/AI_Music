import numpy as np
import pytest

from tools.voice_studio import band_restore


def _tone(freqs, rate=44100, seconds=2.0, peak=0.3):
    t = np.arange(int(rate * seconds)) / rate
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return (signal * peak).astype(np.float64)


def test_complementary_crossover_is_identity_when_inputs_match():
    donor = _tone([220, 3300, 15000])
    out, report = band_restore.restore_high_band(
        donor, 44100, donor, 44100, out_rate=44100)
    np.testing.assert_allclose(out, donor, atol=1e-9)
    assert report["length"] == len(donor)
    assert report["length_adjust_samples"] == 0
    assert report["time_varying_processing"] is False


def test_restores_a_band_limited_24k_vocal():
    donor = _tone([440, 3000, 15000])
    svc = band_restore.resample(donor, 44100, 24000)
    assert band_restore.hf_share(svc, 24000, 12000, 12000) == 0.0
    out, report = band_restore.restore_high_band(
        svc, 24000, donor, 44100, out_rate=44100)
    assert len(out) == len(donor)
    assert report["hf_share_12k_after"] > 10 * report["hf_share_12k_before"]
    assert report["hf_share_12k_after"] > 0.5 * band_restore.hf_share(donor, 44100, 12000, 22050)
    assert band_restore.hf_share(out, 44100, 12000, 22050) > 0.005


def test_length_is_exactly_preserved_at_48k():
    donor = _tone([440, 15000], rate=48000, seconds=1.5)
    svc = band_restore.resample(donor, 48000, 24000)
    out, report = band_restore.restore_high_band(
        svc, 24000, donor, 48000, out_rate=48000)
    assert len(out) == len(donor)
    assert report["out_rate"] == 48000


def test_rejects_a_different_timeline():
    donor = _tone([440], seconds=2.0)
    svc = band_restore.resample(donor, 44100, 24000)[: -4410]
    with pytest.raises(ValueError, match="timelines differ"):
        band_restore.restore_high_band(svc, 24000, donor, 44100, out_rate=44100)


def test_rejects_non_finite_audio():
    donor = _tone([440])
    svc = donor.copy()
    svc[10] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        band_restore.restore_high_band(svc, 44100, donor, 44100, out_rate=44100)


def test_hf_share_is_zero_for_a_pure_low_tone():
    low = _tone([1000], seconds=1.0)
    assert band_restore.hf_share(low, 44100, 11000, 22050) < 1e-6
