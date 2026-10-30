import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality_rebuild_agent import VocalMetrics, cleanup_is_better, normalized_title, passes_final_gate  # noqa: E402


def metrics(**overrides):
    values = dict(artifact_score=12, gate_score=88, gate_decision="pass", harsh_frame_ratio=0.08,
                  burst_count=1, dropout_seconds=0.1, structural_events=1, noise_events=2)
    values.update(overrides)
    return VocalMetrics(**values)


def test_normalized_title_uses_title_metadata(tmp_path):
    wav = tmp_path / "001_89_review_pop_hash.wav"
    wav.touch()
    wav.with_suffix(".txt").write_text("Title: Postcards From Tomorrow\n", encoding="utf-8")
    assert normalized_title(wav) == "postcardsfromtomorrow"


def test_title_falls_back_to_filename_for_mojibake_metadata(tmp_path):
    wav = tmp_path / "001_95_review_guofeng_月下长安_3bb9c8b78094.wav"
    wav.touch()
    wav.with_suffix(".txt").write_text("Title: 鏈堜笅闀垮畨\n", encoding="utf-8")
    assert normalized_title(wav) == "月下长安"


def test_cleanup_requires_noise_improvement_without_regression():
    before = metrics()
    assert cleanup_is_better(before, metrics(harsh_frame_ratio=0.06))
    assert not cleanup_is_better(before, metrics(harsh_frame_ratio=0.06, dropout_seconds=0.3))
    assert not cleanup_is_better(before, metrics())


def test_final_gate_rejects_structural_breaks():
    assert passes_final_gate(metrics(), max_artifact=22, min_gate=78)
    assert not passes_final_gate(metrics(structural_events=6), max_artifact=22, min_gate=78)
    assert not passes_final_gate(metrics(gate_score=70), max_artifact=22, min_gate=78)
    assert not passes_final_gate(metrics(noise_events=9), max_artifact=22, min_gate=78)


def test_humanize_preferred_allows_air_shelf_proxy_inflation():
    from quality_rebuild_agent import humanize_preferred, HUMANIZE_NOISE_CEILING  # noqa: E402
    before = metrics()
    # artifact +2, gate +1, harsh/noise improved: proxy inflation is tolerated
    assert humanize_preferred(before, metrics(artifact_score=14, gate_score=89,
                                              harsh_frame_ratio=0.05, noise_events=1))
    # structural regression is never tolerated
    assert not humanize_preferred(before, metrics(artifact_score=14, gate_score=89,
                                                  structural_events=5))
    # dropout regression is never tolerated
    assert not humanize_preferred(before, metrics(artifact_score=14, gate_score=89,
                                                  dropout_seconds=0.6))
    # gate regression beyond tolerance is rejected
    assert not humanize_preferred(before, metrics(artifact_score=14, gate_score=75))
    assert HUMANIZE_NOISE_CEILING >= 100
