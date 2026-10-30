from argparse import Namespace
import json

import numpy as np
import pytest

from tools.voice_studio import audio,phrase_comp


def test_linear_crossfade_preserves_sample_timeline_and_exact_parent_regions():
    early=np.linspace(-.6,.7,1000,dtype=np.float32)[:,None]
    late=np.linspace(.4,-.3,1000,dtype=np.float32)[:,None]
    mixed=phrase_comp.crossfade(early,late,450,550)
    assert mixed.shape==early.shape
    np.testing.assert_array_equal(mixed[:450],early[:450])
    np.testing.assert_array_equal(mixed[550:],late[550:])
    alpha=np.linspace(0,1,100)[:,None]
    np.testing.assert_allclose(mixed[450:550],early[450:550]*(1-alpha)+late[450:550]*alpha,atol=1e-7)
    assert np.max(np.abs(mixed))<=max(np.max(np.abs(early)),np.max(np.abs(late)))


def test_search_refuses_a_window_if_only_one_parent_is_quiet():
    early=np.zeros((24000,1),dtype=np.float32)
    late=np.ones((24000,1),dtype=np.float32)*.2
    with pytest.raises(phrase_comp.NoSafeCut,match="No shared low-energy"):
        phrase_comp.choose_cut(early,late,24000,center_seconds=.5,radius_seconds=.1)


@pytest.mark.parametrize("start,end",[(0,20),(10,10),(20,10),(80,100),(5,6)])
def test_crossfade_rejects_invalid_or_edge_windows(start,end):
    audio_pair=np.zeros((100,1),dtype=np.float32)
    with pytest.raises(ValueError,match="complete interior"):
        phrase_comp.crossfade(audio_pair,audio_pair,start,end)


def test_comp_rejects_wrong_lengths_without_padding_or_resampling():
    a=np.zeros((1000,1),dtype=np.float32);b=np.zeros((999,1),dtype=np.float32)
    with pytest.raises(ValueError,match="exactly the same frame count"):
        phrase_comp.crossfade(a,b,300,400)
    with pytest.raises(ValueError,match="exactly the same frame count"):
        phrase_comp.choose_cut(a,b,24000,center_seconds=.02,radius_seconds=.005)


def test_cpu_wav_comp_reports_provenance_and_never_claims_inference(tmp_path):
    early=tmp_path/"early.wav";late=tmp_path/"late.wav"
    a=np.ones((24000,1),dtype=np.float32)*.1;b=np.ones((24000,1),dtype=np.float32)*-.15
    a[11000:13000]=0;b[11000:13000]=0
    audio.write(early,a);audio.write(late,b)
    score=tmp_path/"score.json"
    audio.atomic_json(score,[dict(time=[0,1000],duration="1",text="ah",phoneme="en_AA1",note_pitch="60",note_type="2")])
    lyrics=tmp_path/"lyrics.txt";lyrics.write_text("ah")
    args=Namespace(early=early,late=late,early_score=score,late_score=score,lyrics=lyrics,out_dir=tmp_path/"comp",
        sample_rate=24000,expected_seconds=1,center=.5,radius=.05,fade_ms=20,max_rms_db=-34,max_window_peak=.08,
        boundary_note="Synthetic test quiet window")
    result=phrase_comp.comp(args)
    exported,rate=audio.read(args.out_dir/"comp_vocals.wav")
    assert len(exported)==24000 and rate==24000
    assert result["status"]=="awaiting_listening"
    assert result["kind"]=="phrase_comp" and result["inference_performed"] is False and result["gpu_used"] is False
    assert result["parents"]["early"]["sha256"]==audio.digest(early)
    assert result["parents"]["late"]["sha256"]==audio.digest(late)
    assert result["outputs"]["vocals"]["sha256"]==audio.digest(args.out_dir/"comp_vocals.wav")
    assert (args.out_dir/"intended_lyrics.txt").read_text()=="ah"
    assert result["helper_source"]["sha256"]==audio.digest(__import__('pathlib').Path(phrase_comp.__file__))
    assert result["request"]["fade_ms"]==20


def test_nonempty_output_directory_is_rejected_before_reading_inputs_and_preserves_old_files(tmp_path):
    directory=tmp_path/"existing_candidate";directory.mkdir()
    vocals=directory/"comp_vocals.wav";vocals.write_bytes(b"existing audio must survive")
    manifest=directory/"manifest.json";manifest.write_text('{"status":"previous_candidate"}')
    old_hashes={p.name:audio.digest(p) for p in [vocals,manifest]}
    # No parent arguments are provided: the safety guard must run before any input read or write.
    with pytest.raises(FileExistsError,match="use a new output directory"):
        phrase_comp.comp(Namespace(out_dir=directory))
    assert {p.name:audio.digest(p) for p in [vocals,manifest]}==old_hashes
    assert sorted(p.name for p in directory.iterdir())==["comp_vocals.wav","manifest.json"]
