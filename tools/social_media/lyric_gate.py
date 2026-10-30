"""Shared lyric diversity gate for offline batch submit scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DIVERSITY_PATH = PROJECT_ROOT / "services" / "api" / "app" / "services" / "lyric_diversity.py"


def _load_analyze():
    spec = importlib.util.spec_from_file_location("lyric_diversity", _DIVERSITY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load lyric diversity module from {_DIVERSITY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lyric_diversity"] = module
    spec.loader.exec_module(module)
    return module.analyze_lyrics


def check_song_lyrics(title: str, lyrics: str, *, strict: bool = True) -> list[str]:
    analyze_lyrics = _load_analyze()
    report = analyze_lyrics(lyrics)
    if report.passes:
        return []
    issues = [f"[{title}] {issue}" for issue in report.issues]
    if strict:
        raise ValueError("Lyric diversity gate failed:\n- " + "\n- ".join(issues))
    return issues
