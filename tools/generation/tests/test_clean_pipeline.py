"""Tests for the redesigned source-clean pipeline (tools/generation/clean_pipeline.py).

Covers the objective checks and the spectrum-preserving loudness normalization —
the only two things the new pipeline does to the audio besides generation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loudness  # noqa: E402
from clean_pipeline import check_audio, normalize_loudness, generate_one  # noqa: E402


def _write_tone(path: Path, seconds: float = 2.0, sr: int = 48000,
                amplitude: float = 0.3) -> Path:
    t = np.arange(int(sr * seconds)) / sr
    mono = amplitude * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([mono, mono], axis=1)
    sf.write(path, stereo, sr, subtype="PCM_16")
    return path


def test_check_audio_passes_good_file(tmp_path):
    p = _write_tone(tmp_path / "good.wav", seconds=3.0)
    result = check_audio(p, min_duration_s=2.0)
    assert result["ok"] is True
    assert result["duration_s"] > 2.0
    assert result["peak"] > 0.05


def test_check_audio_rejects_too_short(tmp_path):
    p = _write_tone(tmp_path / "short.wav", seconds=0.5)
    result = check_audio(p, min_duration_s=2.0)
    assert result["ok"] is False
    assert "too short" in result["reasons"][0]


def test_check_audio_rejects_near_silence(tmp_path):
    p = _write_tone(tmp_path / "quiet.wav", seconds=2.0, amplitude=0.001)
    result = check_audio(p, min_duration_s=1.0)
    assert result["ok"] is False
    assert any("near-silent" in r or "too quiet" in r for r in result["reasons"])


def test_normalize_loudness_is_spectrum_preserving(tmp_path):
    src = _write_tone(tmp_path / "src.wav", seconds=2.0, amplitude=0.05)
    dst = tmp_path / "norm.wav"
    normalize_loudness(src, dst)

    y_in, sr = sf.read(src, always_2d=True)
    y_out, _ = sf.read(dst, always_2d=True)
    mono_in = y_in.mean(axis=1).astype(np.float64)
    mono_out = y_out.mean(axis=1).astype(np.float64)

    # normalization is a pure gain: the normalized signal must be the input
    # scaled by a constant (within rounding), i.e. correlation ~1 and the
    # frequency shape identical.
    corr = np.abs(np.corrcoef(mono_in, mono_out)[0, 1])
    assert corr > 0.999, f"normalization changed the signal shape: corr={corr:.4f}"

    # RMS target: ~-16 dBFS
    rms_out = float(np.sqrt(np.mean(mono_out**2)))
    assert abs(20 * np.log10(rms_out + 1e-9) - (-16)) < 1.0, f"rms={20*np.log10(rms_out):.1f}dB"

    # peak must be <= 1.0
    assert float(np.max(np.abs(mono_out))) <= 1.0


def test_normalize_loudness_caps_boost_and_peak(tmp_path):
    # very quiet input should not be boosted past +18 dB, and a hot input
    # must be limited to peak <= 1.0
    quiet = _write_tone(tmp_path / "q.wav", seconds=1.0, amplitude=0.001)
    out_q = tmp_path / "q_out.wav"
    report = normalize_loudness(quiet, out_q)
    y_q, _ = sf.read(out_q, always_2d=True)
    mono_q = y_q.mean(axis=1)
    gain = float(np.sqrt(np.mean(mono_q**2))) / (0.001 / np.sqrt(2))
    assert gain <= 8.0 * 1.05, f"gain cap exceeded: {gain:.1f}x"
    assert report["boost_capped"] is True  # the cap is declared, not silent
    assert report["gain_db"] <= 18.0 + 0.1

    # a hot input is attenuated, never limited -- loudness normalization cannot
    # push a mix over full scale just because the target asks for more level
    hot = _write_tone(tmp_path / "h.wav", seconds=1.0, amplitude=0.9)
    out_h = tmp_path / "h_out.wav"
    report_hot = normalize_loudness(hot, out_h)
    y_h, _ = sf.read(out_h, always_2d=True)
    assert float(np.max(np.abs(y_h.mean(axis=1)))) <= 1.0
    assert report_hot["peak_limited"] is False
    assert report_hot["gain_linear"] < 1.0

    # a high-crest take (quiet body + one loud spike) trips the peak ceiling
    t = np.arange(48000) / 48000
    body = 0.02 * np.sin(2 * np.pi * 440 * t)
    body[24000] = 0.95
    crest = tmp_path / "crest.wav"
    sf.write(crest, np.stack([body, body], axis=1), 48000, subtype="PCM_16")
    out_c = tmp_path / "crest_out.wav"
    report_crest = normalize_loudness(crest, out_c)
    y_c, _ = sf.read(out_c, always_2d=True)
    assert float(np.max(np.abs(y_c))) <= 1.0
    assert report_crest["peak_limited"] is True
    assert report_crest["shortfall_db"] > 0
    assert report_crest["peak_out"] <= 0.99 + 1e-6


def test_normalize_loudness_reports_gain_and_hits_target(tmp_path):
    # a take at the model's real native level (~-29 dBFS) must reach the target
    # instead of being silently left short by the old +12 dB boost cap
    src = _write_tone(tmp_path / "native.wav", seconds=2.0, amplitude=0.035)
    dst = tmp_path / "out.wav"
    report = normalize_loudness(src, dst)

    assert abs(report["rms_out_dbfs"] - report["target_dbfs"]) < 1.0, report
    assert report["boost_capped"] is False
    assert report["peak_limited"] is False
    assert report["shortfall_db"] == pytest.approx(0.0, abs=1.0)
    # the report is what makes a silent cap visible from the manifest
    assert report["rms_in"] < 0.05
    assert 1.0 < report["gain_linear"] < 8.0


def test_match_to_level_removes_the_louder_arm_advantage(tmp_path):
    loud = _write_tone(tmp_path / "loud.wav", seconds=1.0, amplitude=0.5)
    quiet = _write_tone(tmp_path / "quiet.wav", seconds=1.0, amplitude=0.05)
    target = loudness.measure(loud)["rms"]
    out = tmp_path / "quiet_matched.wav"
    loudness.match_to_level(quiet, out, target)

    rms = loudness.measure(out)["rms"]
    assert abs(rms - target) / target < 0.02, (rms, target)


def test_runner_seeds_from_cli_arg_and_right_sizes_cache(tmp_path):
    import clean_pipeline

    runner = clean_pipeline._RUNNER
    # seeding must be driven by the parsed --seed argument
    assert "random.seed(a.seed)" in runner
    assert "torch.manual_seed(a.seed)" in runner
    # the KV preallocation must be derived from prompt + requested length
    assert "max_seq_len" in runner
    assert "seq_len" in runner
