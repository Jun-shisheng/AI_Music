"""Tests for the candidate ranking and the adaptive second take.

2026-10-15: the user's verdict on the first English epic batch was "低的时候不够
低，高的时候不够高有点平了" -- the arc terms the ranking had been optimising
(lift, late peak) were already maxed out on that delivery, whose median level sat
only 2.7-6.8 dB under its own peak. The `shape` terms and the stop rule below
encode what that batch scored badly on.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from battle_batch import score_take  # noqa: E402
from yue2_batch_battle import shape_is_good  # noqa: E402

OK = {"ok": True, "reasons": []}


def _arc(**over) -> dict:
    """A take with the shape of the 2026-10-14 delivery: late peak, flat body."""
    arc = {"verdict": "build-up", "lift_db": 15.0, "peak_at": 0.88,
           "body_db": -3.0, "mid_dip_db": -18.0, "onset_mean": 0.06,
           "low_share_peak": 0.08, "quiet_share": 0.12}
    arc.update(over)
    return arc


def test_flat_body_is_not_good_enough_to_stop_at():
    # -3 dB under the peak is what the user heard as flat, even with a textbook
    # build-up arc and a late climax.
    assert shape_is_good(_arc()) is False


def test_dynamic_body_stops_after_one_take():
    assert shape_is_good(_arc(body_db=-9.0)) is True


def test_early_peak_is_never_good_enough():
    assert shape_is_good(_arc(body_db=-12.0, peak_at=0.42)) is False
    assert shape_is_good(_arc(body_db=-12.0, verdict="early peak")) is False


def test_threshold_is_tunable():
    assert shape_is_good(_arc(body_db=-5.0), body_ceiling_db=-4.0) is True


def test_shape_terms_prefer_the_dynamic_take():
    flat = score_take(OK, _arc(body_db=-3.0, onset_mean=0.05,
                               low_share_peak=0.06), {"coverage": 1.0, "ratio": 0.5})
    dynamic = score_take(OK, _arc(body_db=-10.0, onset_mean=0.10,
                                  low_share_peak=0.13, mid_dip_db=-26.0),
                         {"coverage": 1.0, "ratio": 0.5})
    assert dynamic["shape_score"] > flat["shape_score"]
    assert dynamic["total"] > flat["total"]


def test_arc_terms_still_count():
    # A take with a better arc must still win when the shape is equal.
    weak = score_take(OK, _arc(verdict="early peak", peak_at=0.2),
                      {"coverage": 1.0, "ratio": 0.5})
    strong = score_take(OK, _arc(), {"coverage": 1.0, "ratio": 0.5})
    assert strong["total"] > weak["total"]


def test_missing_shape_metrics_do_not_crash_the_ranking():
    # arc dicts written before 2026-10-15 have no body/onset/low fields.
    old = {"verdict": "build-up", "lift_db": 12.0, "peak_at": 0.9}
    score = score_take(OK, old, {"coverage": 1.0, "ratio": 0.4})
    assert score["total"] > 0
    assert shape_is_good(old) is False      # body_db defaults to 0 -> too flat
