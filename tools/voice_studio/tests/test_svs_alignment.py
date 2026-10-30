import copy
from decimal import Decimal

import pytest

from tools.voice_studio import svs_alignment as alignment


def fragmented_chorus():
    pronunciations=dict(Bright="en_B-R-AY1-T",sparks="en_S-P-AA1-R-K-S",rising="en_R-AY1-Z-IH0-NG",I="en_AY1",
        walk="en_W-AO1-K",on="en_AA1-N",Through="en_TH-R-UW1",the="en_DH-AH0",storm="en_S-T-AO1-R-M",
        find="en_F-AY1-N-D",my="en_M-AY1",song="en_S-AO1-NG",SP="<SP>")
    rows=[("Bright",66,2,".348837"),("Bright",63,3,".348837"),("sparks",70,2,".348838"),("sparks",68,3,".348837"),
        ("sparks",63,3,".174419"),("rising",63,2,".174419"),("SP",0,1,".348837"),("rising",70,3,".348837"),
        ("I",63,2,".348837"),("walk",68,2,".348837"),("walk",63,3,".348837"),("SP",0,1,".348837"),
        ("walk",71,3,".174419"),("on",71,2,".174419"),("on",70,3,".348837"),("Through",70,2,".348837"),
        ("the",71,2,".348837"),("storm",70,2,".348837"),("I",71,2,".348837"),("find",68,2,".523256"),
        ("SP",0,1,".523256"),("find",66,3,".174419"),("my",68,2,".348837"),("song",66,2,".348837")]
    return dict(duration=" ".join(r[3] for r in rows),text=" ".join(r[0] if r[0]!="SP" else "<SP>" for r in rows),
        phoneme=" ".join(pronunciations[r[0]] for r in rows),note_pitch=" ".join(str(r[1]) for r in rows),
        note_type=" ".join(str(r[2]) for r in rows))


def test_rest_aware_recipe_preserves_lyrics_pitch_and_timing_with_capacity_improvement():
    source=fragmented_chorus();original=copy.deepcopy(source)
    result,patch=alignment.chorus_v3(source)
    assert source==original
    assert patch["invariants"]["pitch_timeline_identical"]
    assert patch["invariants"]["rest_timeline_identical"]
    assert patch["invariants"]["dictionary_pronunciations_identical"]
    assert sum(map(Decimal,source["duration"].split()))==sum(map(Decimal,result["duration"].split()))
    assert len(patch["before"]["rest_crossing_word_continuations"])==3
    assert patch["after"]["rest_crossing_word_continuations"]==[]
    assert len(patch["before"]["zero_body_english_slots"])==1
    assert patch["after"]["zero_body_english_slots"]==[]
    assert result["alignment_status"]=="approximate_alignment"


def test_recipe_refuses_an_unrecognized_target_instead_of_guessing():
    source=fragmented_chorus();source["text"]=source["text"].replace("rising","different")
    with pytest.raises(ValueError,match="expected the observed"):
        alignment.chorus_v3(source)


def test_invariant_validation_catches_a_pitch_change():
    before=alignment.unpack(fragmented_chorus());after=copy.deepcopy(before)
    after[0]["pitch"]+=1
    with pytest.raises(ValueError,match="pitch or rest"):
        alignment.pitch_invariants(before,after)


def test_invariant_validation_catches_word_order_change():
    before=alignment.unpack(fragmented_chorus());after=copy.deepcopy(before)
    after[0]["text"]="sparks"
    with pytest.raises(ValueError,match="new-word order"):
        alignment.pitch_invariants(before,after)
