import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vocal_resing_agent import _decision, _events_in_segment, _merge_intervals  # noqa: E402


def test_merge_intervals_bridges_short_gaps():
    intervals = [(0.0, 1.0), (1.2, 2.0), (3.0, 4.0)]

    assert _merge_intervals(intervals, max_gap=0.3) == [(0.0, 2.0), (3.0, 4.0)]


def test_events_in_segment_filters_by_timestamp():
    events = [
        {"time_seconds": 0.5, "type": "ok"},
        {"time_seconds": 3.0, "type": "bad"},
    ]

    assert _events_in_segment(events, 0.0, 1.0) == [events[0]]


def test_decision_prefers_partial_then_full_resing():
    assert _decision(8, 0, resing_score=16, full_resing_score=40) == "repair_only"
    assert _decision(20, 0, resing_score=16, full_resing_score=40) == "partial_resing"
    assert _decision(8, 1, resing_score=16, full_resing_score=40) == "partial_resing"
    assert _decision(45, 0, resing_score=16, full_resing_score=40) == "full_resing"
