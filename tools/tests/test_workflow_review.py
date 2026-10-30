import json
from pathlib import Path

from tools.workflow_review import build, ready_render, sha, voice_output


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_changed_completed_render_is_not_ready(tmp_path):
    audio = tmp_path / "renders/baseline/audio.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"original")
    write_json(audio.parent / "state.json", {"status": "complete", "audio_sha256": sha(audio)})
    assert ready_render(tmp_path, "baseline") == audio
    audio.write_bytes(b"different generation")
    assert not ready_render(tmp_path, "baseline").is_file()


def test_failed_or_mutated_voice_output_is_not_presented(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"voice")
    data = {"status": "failed", "outputs": {"vocals": {"path": str(audio), "sha256": sha(audio)}}}
    assert not voice_output(data).is_file()
    data["status"] = "awaiting_listening"
    assert voice_output(data) == audio
    audio.write_bytes(b"changed")
    assert not voice_output(data).is_file()


def test_review_rejects_stale_report_and_exports_sample_identity(tmp_path):
    experiment = tmp_path / "delivery/reference/iron_rain_support"
    write_json(experiment / "manifest.json", {"experiment_identity": "current_generation"})
    files = {}
    for variant in ("baseline", "arranged"):
        native = experiment / f"renders/{variant}/audio.wav"
        native.parent.mkdir(parents=True)
        native.write_bytes(variant.encode())
        write_json(native.parent / "state.json", {"status": "complete", "audio_sha256": sha(native)})
        normalized = experiment / f"listening/{variant}.wav"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_bytes(b"previous version")
        files[variant] = {"listening_file": str(normalized), "source_sha256": sha(native),
                          "listening_sha256": sha(normalized)}
    report = {"experiment_identity": "stale_generation", "loudness_match": {"files": files}}
    write_json(experiment / "report.json", report)
    page = build(tmp_path / "delivery", tmp_path / "workspace")
    output = json.loads((page.parent / "manifest.json").read_text(encoding="utf-8"))
    assert all("renders" in item["source"] for item in output["audio"])
    assert "current_generation" in page.read_text(encoding="utf-8")
    assert "context:" in page.read_text(encoding="utf-8")
    report["experiment_identity"] = "current_generation"
    write_json(experiment / "report.json", report)
    build(tmp_path / "delivery", tmp_path / "workspace")
    output = json.loads((page.parent / "manifest.json").read_text(encoding="utf-8"))
    assert all("listening" in item["source"] for item in output["audio"])
