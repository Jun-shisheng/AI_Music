import json

import numpy as np

from tools.voice_studio import registers


def _three_register_vocal(rate=44100, seconds_each=6.0):
    parts = []
    for f0 in (220.0, 440.0, 880.0):
        t = np.arange(int(rate * seconds_each)) / rate
        part = (0.22 * np.sin(2 * np.pi * f0 * t)
                + 0.08 * np.sin(4 * np.pi * f0 * t)
                + 0.03 * np.sin(6 * np.pi * f0 * t))
        parts.append(part)
    return np.concatenate(parts).astype(np.float32)


def test_extracts_low_mid_and_high_registers(tmp_path):
    import soundfile as sf
    source = tmp_path / "vocal.wav"
    sf.write(str(source), _three_register_vocal(), 44100, subtype="PCM_16")
    manifest = registers.extract_registers(source, tmp_path / "refs",
                                           min_seconds=4, max_seconds=6)
    assert set(manifest["registers"]) == {"low", "mid", "high"}
    medians = {name: entry["median_f0_hz"] for name, entry in manifest["registers"].items()}
    assert medians["low"] < 300
    assert 350 < medians["mid"] < 550
    assert medians["high"] > 700
    assert manifest["register_gap"] is None

    for name, entry in manifest["registers"].items():
        prompt, prompt_rate = sf.read(entry["prompt"]["path"])
        assert prompt_rate == 24000
        assert 3.5 <= len(prompt) / prompt_rate <= 6.5
        contour = np.load(entry["f0"]["path"])
        assert abs(len(contour) - round(len(prompt) / 24000 * 50)) <= 1


def test_select_register_uses_log_f0_distance():
    manifest = {"registers": {
        "low": {"median_f0_hz": 220.0},
        "mid": {"median_f0_hz": 440.0},
        "high": {"median_f0_hz": 880.0},
    }}
    assert registers.select_register(manifest, 800.0) == "high"
    assert registers.select_register(manifest, 460.0) == "mid"
    assert registers.select_register(manifest, 180.0) == "low"


def test_octave_slip_is_rejected():
    contour = np.full(20, 440.0)
    contour[10] = 880.0
    cleaned = registers.reject_octave_slips(contour)
    assert cleaned[10] == 0.0
    assert np.count_nonzero(cleaned) == 19


def test_manifest_hashes_and_schema(tmp_path):
    import soundfile as sf
    source = tmp_path / "vocal.wav"
    sf.write(str(source), _three_register_vocal(seconds_each=4.0), 44100, subtype="PCM_16")
    manifest = registers.extract_registers(source, tmp_path / "refs",
                                           min_seconds=3, max_seconds=4)
    stored = json.loads((tmp_path / "refs" / "registers.json").read_text(encoding="utf-8"))
    assert stored["schema"] == 1
    assert stored["source_sha256"] == manifest["source_sha256"]
    assert len(stored["source_sha256"]) == 64
