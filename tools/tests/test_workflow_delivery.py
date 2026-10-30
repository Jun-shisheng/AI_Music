import json
from pathlib import Path
import re
import wave

import pytest

from tools.workflow_delivery import Collector, collect, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\0\0" * 240)


def setup_job(tmp_path):
    workspace = tmp_path / "workspace"
    folder = workspace / "jobs/relyric_custom_v1"
    output = folder / "converted_vocals.wav"
    wav(output)
    (folder / "intended_lyrics.txt").write_text("Different lyrics for this exact sample", encoding="utf-8")
    write(folder / "manifest.json", {"status": "awaiting_listening", "inputs": {"command": "relyric"},
          "outputs": {"vocals": {"path": str(output), "sha256": sha(output)}}})
    return workspace, output


def test_copy_rejects_bad_hash_and_escaping_destination(tmp_path):
    source = tmp_path / "source.wav"
    wav(source)
    collector = Collector(tmp_path / "delivery")
    with pytest.raises(ValueError, match="SHA mismatch"):
        collector.copy(source, Path("audio.wav"), "incorrect")
    assert not (tmp_path / "delivery/audio.wav").exists()
    with pytest.raises(ValueError, match="escapes"):
        collector.copy(source, Path("../outside.wav"))


def test_daily_delivery_is_repeatable_and_self_contained(tmp_path):
    workspace, _ = setup_job(tmp_path)
    destination = tmp_path / "2026-10-19"
    page = collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    first = json.loads((destination / "package_manifest.json").read_text(encoding="utf-8"))
    assert "Different lyrics for this exact sample" in page.read_text(encoding="utf-8")
    for relative in re.findall(r'<audio[^>]+src="([^"]+)"', page.read_text(encoding="utf-8")):
        assert not Path(relative).is_absolute()
        assert (destination / relative).is_file()
    collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    second = json.loads((destination / "package_manifest.json").read_text(encoding="utf-8"))
    assert first["audio"] == second["audio"]
    assert first["archive_files"] == second["archive_files"]


def test_changed_archive_is_preserved_and_reported(tmp_path):
    workspace, _ = setup_job(tmp_path)
    destination = tmp_path / "2026-10-19"
    collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    manifest = json.loads((destination / "package_manifest.json").read_text(encoding="utf-8"))
    archived = next(key for key in manifest["archive_files"] if key.endswith("converted_vocals.wav"))
    (destination / archived).write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive is missing or changed"):
        collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    assert (destination / archived).read_bytes() == b"changed"


def test_unfinished_or_changed_outputs_are_not_presented(tmp_path):
    workspace, output = setup_job(tmp_path)
    output.write_bytes(b"unverified replacement")
    destination = tmp_path / "2026-10-19"
    page = collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    manifest = json.loads((destination / "package_manifest.json").read_text(encoding="utf-8"))
    assert not manifest["audio"]
    assert manifest["warnings"]
    assert "音频摘要失配" in page.read_text(encoding="utf-8")


def test_completed_first_arm_without_loudness_report_uses_native(tmp_path):
    destination = tmp_path / "2026-10-19"
    experiment = destination / "experiments/reference/new_pair"
    audio = experiment / "renders/baseline/audio.wav"
    wav(audio)
    write(audio.parent / "state.json", {"status": "complete", "audio_sha256": sha(audio)})
    write(experiment / "manifest.json", {"experiment_identity": "new-pair"})
    write(experiment / "report.json", {"experiment_identity": "new-pair", "loudness_match": None})
    page = collect(destination, tmp_path / "workspace", tmp_path / "missing_legacy", tmp_path / "repo")
    assert "基准版（原始响度）" in page.read_text(encoding="utf-8")
    data = json.loads((destination / "package_manifest.json").read_text(encoding="utf-8"))
    assert len(data["audio"]) == 1
    assert data["audio"][0]["sha256"] == sha(audio)


def test_failed_job_residual_audio_is_diagnostic_only(tmp_path):
    workspace, output = setup_job(tmp_path)
    path = output.parent / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    write(path, manifest)
    destination = tmp_path / "2026-10-19"
    collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    package = json.loads((destination / "package_manifest.json").read_text(encoding="utf-8"))
    assert package["audio"] == []
    assert len(package["failed_job_diagnostics"]) == 1
    assert (destination / package["failed_job_diagnostics"][0] / output.name).is_file()


def test_completed_reference_with_duration_drift_is_visibly_flagged(tmp_path):
    destination = tmp_path / "2026-10-19"
    experiment = destination / "experiments/reference/full_harmony"
    audio = experiment / "renders/arranged/audio.wav"
    wav(audio)
    write(audio.parent / "state.json", {"status": "complete", "audio_sha256": sha(audio)})
    write(experiment / "manifest.json", {"experiment_identity": "harmony-v1"})
    write(experiment / "report.json", {"experiment_identity": "harmony-v1", "loudness_match": None,
          "metrics": {"arranged": {"duration_validation": {"technical_review_required": True,
          "actual_duration_s": 32.918667, "relative_error_percent": 47.445}}}})
    page = collect(destination, tmp_path / "workspace", tmp_path / "missing_legacy", tmp_path / "repo")
    text = page.read_text(encoding="utf-8")
    assert "实际32.919秒" in text and "+47.4%" in text and "未通过时长检查" in text
    assert '<audio controls' in text


def test_updated_legacy_report_gets_new_snapshot(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    source = legacy / "report.json"
    write(source, {"revision": 1})
    destination = tmp_path / "2026-10-19"
    collect(destination, tmp_path / "workspace", legacy, tmp_path / "repo")
    original = destination / "archive/initial_quality_upgrade/report.json"
    before = original.read_bytes()
    write(source, {"revision": 2})
    collect(destination, tmp_path / "workspace", legacy, tmp_path / "repo")
    package = json.loads((destination / "package_manifest.json").read_text(encoding="utf-8"))
    assert original.read_bytes() == before
    assert package["latest_legacy_snapshot"] != "archive/initial_quality_upgrade"
    assert json.loads((destination / package["latest_legacy_snapshot"] / "report.json").read_text())["revision"] == 2


def test_qa_is_attached_only_when_output_hashes_are_current(tmp_path):
    workspace, audio = setup_job(tmp_path)
    destination = tmp_path / "2026-10-19"
    folder = destination / "qa/sample"
    request = folder / "request.json"
    result = folder / "qa.json"
    write(request, {"audio": {"sha256": sha(audio)}})
    write(result, {"status": "complete", "request_identity": "qa-v1", "target_lines": ["Different lyrics for this exact sample"], "comparison": {"word_edits": {"insertion": 2}}})
    write(folder / "qa_manifest.json", {"status": "complete", "request_identity": "qa-v1",
          "outputs": {"qa.json": sha(result), "request.json": sha(request)}})
    collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    package = json.loads((destination / "package_manifest.json").read_text())
    assert package["audio"][0]["qa_request_identity"] == "qa-v1"
    write(result, {"status": "complete", "request_identity": "qa-v1", "comparison": {"word_edits": {}}})
    collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    package = json.loads((destination / "package_manifest.json").read_text())
    assert "qa_request_identity" not in package["audio"][0]
    write(result, {"status": "complete", "request_identity": "qa-v2", "target_lines": ["Unrelated lyrics"], "comparison": {"word_edits": {}}})
    write(folder / "qa_manifest.json", {"status": "complete", "request_identity": "qa-v2",
          "outputs": {"qa.json": sha(result), "request.json": sha(request)}})
    collect(destination, workspace, tmp_path / "missing_legacy", tmp_path / "repo")
    package = json.loads((destination / "package_manifest.json").read_text())
    assert "qa_request_identity" not in package["audio"][0]


def test_controlled_assets_are_downloads_with_verified_shared_voice(tmp_path):
    destination = tmp_path / "2026-10-19"
    experiment = destination / "experiments/reference/midi_pair"
    voice = experiment / "source/fixed_vocal.wav"
    wav(voice)
    variants = {}
    for variant in ["sparse", "expanded"]:
        variants[variant] = {"vocal_reference": {"path": "source/fixed_vocal.wav", "sha256": sha(voice)}}
        for key, suffix in [("midi", "mid"), ("events", "json")]:
            path = experiment / f"{variant}.{suffix}"
            path.write_bytes(b"test artifact")
            variants[variant][key] = {"path": path.name, "sha256": sha(path)}
    write(experiment / "manifest.json", {"kind": "controlled_arrangement_prepare", "status": "prepared_not_rendered", "variants": variants})
    page = collect(destination, tmp_path / "workspace", tmp_path / "missing_legacy", tmp_path / "repo")
    assert "可控编曲 · MIDI分轨素材" in page.read_text(encoding="utf-8")
    assert "尚未渲染伴奏或混音" in page.read_text(encoding="utf-8")
    package = json.loads((destination / "package_manifest.json").read_text())
    assert package["audio"] == []
    variants["expanded"]["vocal_reference"]["sha256"] = "incorrect"
    write(experiment / "manifest.json", {"kind": "controlled_arrangement_prepare", "status": "prepared_not_rendered", "variants": variants})
    page = collect(destination, tmp_path / "workspace", tmp_path / "missing_legacy", tmp_path / "repo")
    assert "可控编曲 · MIDI分轨素材" not in page.read_text(encoding="utf-8")
    assert "素材摘要失配" in page.read_text(encoding="utf-8")
