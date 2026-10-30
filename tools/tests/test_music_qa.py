import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

from tools import music_qa as qa


def test_insertions_and_repeated_transcript_never_pass_exact_text():
    expected = ["Bright sparks rising", "I will not retreat"]
    for text in ["Bright sparks rising oh I will not retreat", "Bright sparks rising I will not retreat " * 2]:
        result = qa.compare_text(expected, text)
        assert result["approximate_lines_recovered"] == 2
        assert result["word_edits"]["insertion"] > 0
        assert result["asr_normalized_text_exact_match"] is False
        assert result["content_status"] == "asr_differences_require_review"


def test_one_occurrence_of_repeated_lyric_cannot_recover_two_lines():
    result = qa.compare_text(["I walk on", "I walk on"], "I walk on")
    assert result["approximate_lines_recovered"] == 1
    assert result["word_edits"]["deletion"] == 3
    assert not result["asr_normalized_text_exact_match"]


def test_normalization_and_empty_transcript_keep_real_differences():
    assert qa.compare_text(["I WILL, not retreat!"], "i will not retreat")["asr_normalized_text_exact_match"]
    result = qa.compare_text(["I will not retreat"], "")
    assert result["word_edits"] == {"substitution": 0, "deletion": 4, "insertion": 0}
    assert result["word_coverage"] == 0
    assert result["approximate_lines_recovered"] == 0
    assert qa.tokens("风雨，我前行") == ["风", "雨", "我", "前", "行"]


def test_known_substitution_deletion_and_insertion_counts():
    result = qa.compare_text(["bright sparks rising I walk on"], "right sparks rising walk on again")
    assert result["word_edits"] == {"substitution": 1, "deletion": 1, "insertion": 1}


def test_character_similarity_disables_autojunk_for_long_repeated_text():
    target = "bright sparks rising " * 40
    result = qa.compare_text([target], target + "extra")
    assert result["normalized_character_similarity"] > .99
    assert not result["asr_normalized_text_exact_match"]


def test_only_section_headings_are_not_valid_lyrics():
    with pytest.raises(ValueError, match="no words"):
        qa.lyric_lines("[Chorus]\n\n[Verse 2]")
    assert qa.lyric_lines("\ufeff[Chorus]\nHello world\n[Outro] Stay") == ["Hello world", "Stay"]


def test_native_peak_does_not_disappear_in_antiphase_stereo(tmp_path):
    path = tmp_path / "opposite.wav"
    sf.write(path, np.tile([.9, -.9], (8000, 1)), 8000, subtype="FLOAT")
    result = qa.native_audio(path)
    assert result["valid"]
    assert result["native_sample_peak"] == pytest.approx(.9)
    assert result["rms"] == pytest.approx(.9)


@pytest.mark.parametrize("kind", ["empty", "silent", "nonfinite"])
def test_invalid_audio_is_reported_without_inventing_content(tmp_path, kind):
    path = tmp_path / f"{kind}.wav"
    signal = np.zeros(8000) if kind != "empty" else np.empty(0)
    if kind == "nonfinite":
        signal[42] = np.nan
    sf.write(path, signal, 8000, subtype="FLOAT")
    result = qa.native_audio(path)
    assert not result["valid"]
    assert result["errors"]
    json.dumps(result, allow_nan=False)


def args_and_request(tmp_path, monkeypatch):
    audio = tmp_path / "source.wav"
    sf.write(audio, np.sin(np.arange(8000) * .2) * .1, 8000)
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Chorus]\nBright sparks rising\n", encoding="utf-8")
    args = qa.parser().parse_args(["--audio", str(audio), "--lyrics", str(lyrics), "--out-dir", str(tmp_path / "qa")])
    request = dict(audio=dict(path=str(audio), sha256=qa.sha(audio)),
                   lyrics=dict(path=str(lyrics), sha256=qa.sha(lyrics), target_lines=["Bright sparks rising"]),
                   model=dict(files={}), parameters=dict(language="en"))
    monkeypatch.setattr(qa, "make_request", lambda args: request)
    return args, request


def test_invalid_audio_never_starts_asr_and_writes_failed_manifest(tmp_path, monkeypatch):
    args, _ = args_and_request(tmp_path, monkeypatch)
    calls = []
    def worker(python, kind, spec, directory):
        calls.append(kind)
        assert kind == "audio"
        return dict(valid=False, errors=["Audio has zero frames"], duration_seconds=0,
                    sample_rate=8000, channels=1, native_sample_peak=0, nonfinite_samples=0)
    monkeypatch.setattr(qa, "run_worker", worker)
    result = qa.execute(args)
    assert result["status"] == "failed"
    assert calls == ["audio"]
    assert qa.read_json(args.out_dir / "qa_manifest.json")["status"] == "failed"
    assert "comparison" not in result


def test_valid_cache_is_reused_but_changed_report_is_recomputed(tmp_path, monkeypatch):
    args, request = args_and_request(tmp_path, monkeypatch)
    calls = []
    def worker(python, kind, spec, directory):
        calls.append(kind)
        if kind == "audio":
            data = dict(valid=True, duration_seconds=1, sample_rate=8000, channels=1,
                        native_sample_peak=.1, nonfinite_samples=0)
            qa.write_json(directory / "audio.json", data)
        else:
            assert "lyrics" not in spec and "initial_prompt" not in spec
            data = dict(text="Bright sparks rising", segments=[])
            qa.write_json(directory / "transcript.json", data)
        return data
    monkeypatch.setattr(qa, "run_worker", worker)
    assert qa.execute(args)["status"] == "complete"
    assert qa.execute(args)["status"] == "complete"
    assert len(calls) == 2
    (args.out_dir / "transcript.json").write_text("{}")
    assert qa.execute(args)["status"] == "complete"
    assert len(calls) == 4
    request["audio"]["sha256"] = "different"
    with pytest.raises(ValueError, match="different inputs"):
        qa.execute(args)


def test_worker_never_supplies_target_prompt_or_gpu(monkeypatch):
    captured = {}
    class FakeModel:
        def __init__(self, *args, **kwargs):
            captured["model"] = kwargs
        def transcribe(self, path, **kwargs):
            captured["transcribe"] = kwargs
            return iter([]), type("Info", (), dict(language="en", language_probability=1.))()
    monkeypatch.setitem(sys.modules, "faster_whisper", type("Module", (), dict(WhisperModel=FakeModel)))
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test")
    qa.transcribe(dict(audio="audio.wav", model="local", language="en"))
    assert captured["model"]["device"] == "cpu"
    assert captured["model"]["cpu_threads"] == 2
    assert captured["model"]["compute_type"] == "int8"
    assert captured["model"]["local_files_only"] is True
    assert not {"initial_prompt", "prefix", "hotwords"} & captured["transcribe"].keys()


def test_input_changed_during_asr_never_gets_complete_state(tmp_path, monkeypatch):
    args, _ = args_and_request(tmp_path, monkeypatch)
    def worker(python, kind, spec, directory):
        if kind == "audio":
            return dict(valid=True, duration_seconds=1, sample_rate=8000, channels=1,
                        native_sample_peak=.1, nonfinite_samples=0)
        args.audio.write_bytes(b"a different generation")
        return dict(text="Bright sparks rising", segments=[])
    monkeypatch.setattr(qa, "run_worker", worker)
    result = qa.execute(args)
    assert result["status"] == "failed"
    assert "changed during QA" in result["error"]
    assert qa.read_json(args.out_dir / "qa_manifest.json")["status"] == "failed"
