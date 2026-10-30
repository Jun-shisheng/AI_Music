import numpy as np

from tools.voice_studio import audio


def _signal(seconds=31.0, rate=audio.RATE, silence=None):
    t = np.arange(int(rate * seconds)) / rate
    value = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    if silence:
        start, end = silence
        value[int(start * rate):int(end * rate)] = 0
    return value


def _f0(value, rate=audio.RATE, voiced_hz=300.0):
    frames = int(np.ceil(len(value) / rate * 50))
    contour = np.full(frames, voiced_hz, dtype=np.float64)
    levels = audio.rms(value, rate // 50)
    contour[:min(frames, len(levels))][levels[:min(frames, len(levels))] < 1e-4] = 0
    return contour


def _assert_merge_identity(value, entries):
    rebuilt = audio.merge([(e, value[e["context_start"]:e["context_end"]]) for e in entries],
                          len(value))
    np.testing.assert_allclose(value, rebuilt, atol=1e-6)


def test_cuts_at_a_long_rest_and_preserves_the_timeline():
    value = _signal(silence=(9.0, 11.5))
    entries = audio.voiced_segments(value, audio.RATE, _f0(value), min_seconds=6,
                                    max_seconds=12, context_seconds=.2)
    assert len(entries) >= 3
    rest_cuts = [e for e in entries if e["cut_reason"] in {"rest", "low_energy"}]
    assert rest_cuts, entries
    cut_seconds = [e["end"] / audio.RATE for e in entries[:-1]]
    assert any(9.0 <= cut <= 11.5 for cut in cut_seconds), cut_seconds
    _assert_merge_identity(value, entries)


def test_sustained_tone_marks_forced_cuts_instead_of_hiding_them():
    value = _signal()
    entries = audio.voiced_segments(value, audio.RATE, _f0(value), min_seconds=6,
                                    max_seconds=12, context_seconds=.2)
    assert len(entries) > 1
    assert all(e["cut_reason"] in {"forced_voiced_cut", "tail_rebalanced"} for e in entries[:-1])
    assert any(e["cut_reason"] == "forced_voiced_cut" for e in entries[:-1])
    assert all(e["forced"] for e in entries[:-1])
    _assert_merge_identity(value, entries)


def test_chunk_lengths_and_tail_are_bounded():
    value = _signal(seconds=20.0)
    entries = audio.voiced_segments(value, audio.RATE, _f0(value), min_seconds=6,
                                    max_seconds=12, context_seconds=.2)
    for entry in entries:
        assert 6.0 <= entry["chunk_seconds"] <= 15.0 + 1e-6, entry
    assert entries[0]["start"] == 0
    assert entries[-1]["end"] == len(value)
    assert all(a["end"] == b["start"] for a, b in zip(entries, entries[1:]))
    _assert_merge_identity(value, entries)


def test_short_tail_is_merged_or_rebalanced():
    value = _signal(seconds=13.0)
    entries = audio.voiced_segments(value, audio.RATE, _f0(value), min_seconds=6,
                                    max_seconds=12, context_seconds=.2)
    assert entries[-1]["chunk_seconds"] >= 6.0
    _assert_merge_identity(value, entries)


def test_canonical_at_keeps_the_requested_rate(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    audio.write(source, _signal(seconds=1.0), audio.RATE)
    audio.canonical_at(source, target, rate=44100)
    data, rate = audio.read(target)
    assert rate == 44100
    assert abs(len(data) / rate - 1.0) < 0.001
