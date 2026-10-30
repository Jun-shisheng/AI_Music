import json
from pathlib import Path

import numpy as np
import pytest

from tools.voice_studio import audio,svs_score


def score(tmp_path:Path)->Path:
    path=tmp_path/"score.abc"
    path.write_text('X:1\nM:4/4\nL:1/4\nQ:1/4=60\nK:D#m\n% chorus\nV: Vocal\nf a =a a|a4|\nV: Ins\nZ2|\n')
    return path


def test_abc_absolute_pitch_applies_key_and_bar_accidental_reset(tmp_path):
    bars,clock=svs_score.abc_notes(score(tmp_path))
    assert [n["pitch"] for bar in bars for n in bar]==[78,82,81,81,82]
    assert clock["duration_seconds"]==8


def test_unknown_words_are_rejected_instead_of_invented():
    with pytest.raises(ValueError,match="not in the pinned CMU"):
        svs_score.phones("inventedzz",{},set())


def test_target_octave_adaptation_keeps_duration_and_pitch_classes(tmp_path):
    bars,_=svs_score.abc_notes(score(tmp_path))
    dictionary={"bright":[["B","R","AY1","T"]],"on":[["AA1","N"]]}
    phoneset={"en_"+phone for pron in dictionary.values() for phone in pron[0]}
    native,_=svs_score.target_metadata(bars,["Bright on"],dictionary,phoneset,0)
    lower,report=svs_score.target_metadata(bars,["Bright on"],dictionary,phoneset,-12)
    assert native["duration"]==lower["duration"]
    assert native["phoneme"]==lower["phoneme"]
    assert all(int(a)-int(b)==12 for a,b in zip(native["note_pitch"].split(),lower["note_pitch"].split()))
    assert report["absolute_pitch_preserved"] is False
    assert lower["alignment_status"]=="approximate_alignment"


def test_prompt_sustain_inference_is_explicit_and_covers_full_clip(tmp_path):
    prompt=tmp_path/"prompt.wav"
    contour=tmp_path/"prompt_f0.npy"
    audio.write(prompt,np.ones(audio.RATE*3)*.1)
    f0=np.zeros(150);f0[:60]=220;f0[75:100]=220
    np.save(contour,f0)
    asr=dict(prompt_sha256=audio.digest(prompt),segments=[dict(words=[
        dict(word="bright",start=0,end=.5,probability=.99),dict(word="on",start=1.5,end=2,probability=.99)])])
    dictionary={"bright":[["B","R","AY1","T"]],"on":[["AA1","N"]]}
    phoneset={"en_"+phone for pron in dictionary.values() for phone in pron[0]}
    metadata,report=svs_score.prompt_metadata(prompt,contour,asr,dictionary,phoneset)
    assert abs(sum(map(float,metadata["duration"].split()))-3)<1e-6
    assert report["forced_alignment"] is False
    extension=next(x for x in report["uncertainties"] if x["issue"]=="inferred_sustain_beyond_ASR_word_end")
    assert extension["inferred_end"]==1.2
    assert extension["human_verified"] is False
