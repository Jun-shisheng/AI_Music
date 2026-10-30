"""Tests for batch_tuned.py input ordering (priority regex)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_tuned import order_pairs  # noqa: E402

EN = "^1[1-9]_|^20_"


def _slugs(rows):
    return [r[0] for r in rows]


def test_no_priority_keeps_order():
    rows = [("01_a", 0), ("11_b", 1), ("02_c", 2)]
    assert order_pairs(rows, None) == rows


def test_priority_moves_matches_to_front_stably():
    rows = [("09_a", 0), ("10_b", 1), ("11_c", 2), ("12_d", 3), ("20_e", 4)]
    assert _slugs(order_pairs(rows, EN)) == ["11_c", "12_d", "20_e", "09_a", "10_b"]


def test_priority_does_not_drop_or_duplicate():
    rows = [(f"{i:02d}_x", i) for i in range(1, 21)]
    out = order_pairs(rows, EN)
    assert len(out) == len(rows)
    assert sorted(_slugs(out)) == sorted(_slugs(rows))


def test_priority_payload_survives():
    rows = [("10_b", 1), ("13_c", 2)]
    out = order_pairs(rows, EN)
    assert out[0] == ("13_c", 2)
