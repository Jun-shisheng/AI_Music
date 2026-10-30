from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.reference_studio import plan
from tools.reference_studio.audio import (blueprint, decode, descriptors, loudness_match,
                                          native_file_levels, validate_annotations)
from tools.reference_studio.mapping import propose_mapping
from tools.reference_studio.workflow import (checked_manifest, digest, dispatch,
                                             duration_validation, prepare, render_commands, report,
                                             rendering_config, SEMANTIC_DEFAULTS, write_json)
from tools.reference_studio.reuse import reuse_renders
import tools.reference_studio.reuse as reuse_module
from tools.reference_studio.__main__ import main as studio_main
from tools.reference_studio import render_worker


ABC = '''X:1
M:4/4
L:1/16
Q:1/4=120
V: Vocal clef=treble
V: Ins clef=treble
K:C
% intro
V: Vocal
"C"z16|
V: Ins
Z|
% verse
V: Vocal
"Am"c4d4e4z4|"F"f8e4z4|"C"e4d4c4z4|"G"d8z8|
V: Ins
Z4|
% chorus
V: Vocal
"C"g2g2e2e2g2g2a2a2|"G"g2g2d2d2d4z4|"Am"e4e4g4z4|"F"a8z8|
V: Ins
Z4|
'''
LYRICS = '''[Intro]
Intro words
[Verse 1]
First verse words
[Pre-Chorus 1]
Build to the chorus
[Chorus]
First chorus words
[Verse 2]
Second verse words
[Pre-Chorus 2]
Second build
[Chorus 2]
Second chorus words
'''


def source_files(tmp_path):
    files = [tmp_path / "source.abc", tmp_path / "lyrics.txt", tmp_path / "style.txt"]
    for path, text in zip(files, [ABC, LYRICS, "mature female vocal, orchestral rock"]):
        path.write_text(text, encoding="utf-8")
    return files


def make_experiment(tmp_path):
    files = source_files(tmp_path)
    out = tmp_path / "experiment"
    manifest = prepare(*files, out, allow_long=True)
    return out, manifest


@pytest.mark.parametrize("mode", ["sparse", "full", "arranged"])
def test_instrument_experiment_preserves_vocal_harmony_and_time(mode):
    excerpt, info = plan.select_excerpt(ABC)
    edited, changes = plan.edit_instrumental(excerpt, mode)
    original_h, original_s = plan.parse(excerpt)
    new_h, new_s = plan.parse(edited)
    assert original_h == new_h
    assert [c.vocal for s in original_s for c in s.chunks] == [c.vocal for s in new_s for c in s.chunks]
    assert changes["vocal_header_unchanged"]
    assert plan.CHORD.findall(edited) == plan.CHORD.findall(excerpt)
    for sec in new_s:
        for chunk in sec.chunks:
            assert len(plan.bars(chunk.ins, Fraction(16))) == len(plan.bars(chunk.vocal, Fraction(16)))
    assert info["bars"] == 8
    assert info["duration_s"] == 16


def test_sparse_full_and_arranged_have_a_real_symbolic_difference():
    excerpt, _ = plan.select_excerpt(ABC)
    sparse, _ = plan.edit_instrumental(excerpt, "sparse")
    full, _ = plan.edit_instrumental(excerpt, "full")
    arranged, changes = plan.edit_instrumental(excerpt, "arranged")
    assert sparse != full != arranged
    assert [c["mode"] for c in changes["changes"]] == ["sparse", "full"]


def test_support_pattern_uses_only_one_or_two_chord_tones_with_half_bar_space():
    excerpt, _ = plan.select_excerpt(ABC)
    support, changes = plan.edit_instrumental(excerpt, "arranged", family="support")
    busy, _ = plan.edit_instrumental(excerpt, "arranged")
    header, sections = plan.parse(support)
    assert support != busy
    for si, section in enumerate(sections):
        for chunk in section.chunks:
            for bar in plan.bars(chunk.ins, Fraction(16)):
                tokens = [m[0] for m in plan.TOKEN.finditer(bar)]
                notes = [plan.NOTE.fullmatch(token) for token in tokens if plan.NOTE.fullmatch(token)]
                assert len(notes) == (1 if si == 0 else 2)
                assert sum(plan.length(t[1:]) for t in tokens if t.startswith("z")) == 8
    assert plan.vocal_identity(*plan.parse(excerpt)) == plan.vocal_identity(header, sections)
    assert changes["instrument_pattern_family"] == "support"


def test_support_prepare_changes_only_arranged_and_preserves_baseline(tmp_path):
    files = source_files(tmp_path)
    original, candidate = tmp_path/"original", tmp_path/"support"
    prepare(*files, original, allow_long=True)
    manifest = prepare(*files, candidate, allow_long=True, arranged_pattern="support")
    for variant in ("baseline", "sparse", "full"):
        assert (original/"plans"/f"{variant}.abc").read_bytes() == (candidate/"plans"/f"{variant}.abc").read_bytes()
    assert (original/"plans"/"arranged.abc").read_bytes() != (candidate/"plans"/"arranged.abc").read_bytes()
    assert manifest["arranged_pattern"] == "support"


def test_melody_conversion_removes_music_chords_but_preserves_notes_headers_and_time():
    source = ABC.replace("V: Vocal clef=treble", 'V: Vocal clef=treble name="Vocal Melody"')
    converted, record = plan.strip_music_chords(source)
    before_header, before_sections = plan.parse(source)
    after_header, after_sections = plan.parse(converted)
    assert before_header == after_header
    assert 'name="Vocal Melody"' in converted
    for before, after in zip(before_sections, after_sections):
        for a, b in zip(before.chunks, after.chunks):
            for voice in ("vocal", "ins"):
                assert getattr(b, voice) == [plan.CHORD.sub("", line) for line in getattr(a, voice)]
                assert len(plan.bars(getattr(a, voice), Fraction(16))) == len(plan.bars(getattr(b, voice), Fraction(16)))
    assert record["removed_chord_symbols"] == 9
    assert record["note_rest_tie_bar_text_unchanged"]


def test_cli_cot_cfg_flow_through_manifest_and_actual_worker_call(tmp_path, monkeypatch):
    files = source_files(tmp_path)
    out = tmp_path/"melody_cfg"
    result = studio_main(["prepare", "--plan", str(files[0]), "--lyrics", str(files[1]),
                          "--style", str(files[2]), "--out-dir", str(out), "--allow-long",
                          "--cot", "melody", "--cfg-scale", "1.5", "--semantic-max-tokens", "1200"])
    assert result == 0
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["rendering"] == {"cot": "melody", "cfg_scale": 1.5, "semantic_max_tokens": 1200}
    assert manifest["cot_conversion"]["status"] == "applied"
    assert not plan.CHORD.search("\n".join(line for s in plan.parse((out/"plans"/"baseline.abc").read_text())[1] for c in s.chunks for line in c.vocal))
    observed = {}

    class FakeSong:
        truncated = {"abc": False, "semantic": False}
        timing = {}

        def save(self, path):
            Path(path).write_bytes(b"test audio stub")

        def save_artifacts(self, path):
            Path(path).mkdir()

    class FakePipe:
        def __init__(self, *args, **kwargs):
            observed["constructor"] = kwargs

        def __call__(self, **kwargs):
            observed["request"] = kwargs
            return FakeSong()

    fake_package = types.ModuleType("yue2")
    fake_package.__path__ = []
    fake_package.YuE2Pipeline = FakePipe
    fake_nar = types.ModuleType("yue2.nar")
    fake_nar.synthesize = lambda *a, **kw: None
    fake_package.nar = fake_nar
    monkeypatch.setitem(sys.modules, "yue2", fake_package)
    monkeypatch.setitem(sys.modules, "yue2.nar", fake_nar)
    monkeypatch.setattr(sys, "argv", ["render_worker.py", "--experiment", str(out), "--variant", "baseline", "--offload-ar"])
    assert render_worker.main() == 0
    assert observed["request"]["cot"] == "melody"
    assert observed["request"]["cfg_scale"] == 1.5
    assert observed["request"]["semantic_sampling"] == {"max_tokens": 1200}
    assert observed["request"]["seed"] == 1338
    saved = json.loads((out/"renders"/"baseline"/"result.json").read_text())
    assert saved["rendering"] == manifest["rendering"]


def test_default_old_manifest_still_calls_full_without_cfg_override(tmp_path):
    out, manifest = make_experiment(tmp_path)
    manifest.pop("rendering")
    kwargs = render_worker.generation_kwargs(out, manifest, "baseline")
    assert kwargs["cot"] == "full" and "cfg_scale" not in kwargs
    assert "semantic_sampling" not in kwargs


def test_nondefault_cot_cfg_are_members_of_prepared_identity(tmp_path):
    files = source_files(tmp_path)
    ids = set()
    for index, (cot, cfg) in enumerate([("full", None), ("full", 1.5), ("melody", 1.5), ("melody", 2.0)]):
        manifest = prepare(*files, tmp_path/f"case{index}", allow_long=True, cot=cot, cfg_scale=cfg)
        ids.add(manifest["experiment_identity"])
    assert len(ids) == 4


def test_semantic_ceiling_is_in_identity_and_leaves_all_other_sampling_at_native_default(tmp_path):
    files = source_files(tmp_path)
    first = prepare(*files, tmp_path/"cap1200", allow_long=True, semantic_max_tokens=1200)
    second = prepare(*files, tmp_path/"cap2400", allow_long=True, semantic_max_tokens=2400)
    assert first["experiment_identity"] != second["experiment_identity"]
    kwargs = render_worker.generation_kwargs(tmp_path/"cap1200", first, "baseline")
    assert kwargs["semantic_sampling"] == {"max_tokens": 1200}
    with pytest.raises(ValueError, match=">= 200"):
        prepare(*files, tmp_path/"invalid", allow_long=True, semantic_max_tokens=199)


def test_duration_error_above_ten_percent_requires_technical_review():
    assert not duration_validation(66, 60)["technical_review_required"]
    assert duration_validation(66.01, 60)["technical_review_required"]
    assert duration_validation(53.9, 60)["technical_review_required"]
    actual = duration_validation(81.5586666667, 58.605)
    assert 39.16 < actual["relative_error_percent"] < 39.18
    assert not actual["symbolic_section_metrics_allowed"]


def test_blueprint_manual_mapping_overrides_proposed_instrument_pattern():
    excerpt, _ = plan.select_excerpt(ABC)
    bp = {"sections": [{"name": "reference_lift", "target": {"plan_section": "verse", "ins_pattern": "full"}}]}
    _, changes = plan.edit_instrumental(excerpt, "arranged", bp)
    assert changes["changes"][0]["mode"] == "full"


def measured_blueprint(first_db=-24, second_db=-10, first_rate=.4, second_rate=2):
    return {"annotation_status": "draft_acoustic_boundaries_unreviewed", "sections": [
        {"name": "section_07", "start_s": 20, "end_s": 36,
         "annotation_status": "unreviewed", "role_observation_status": "unknown_no_instrument_detector",
         "measured_mix": {"rms_dbfs": first_db, "mix_transient_events_per_s": first_rate},
         "target": {"ins_pattern": "sparse"}},
        {"name": "section_08", "start_s": 36, "end_s": 52,
         "annotation_status": "unreviewed", "role_observation_status": "unknown_no_instrument_detector",
         "measured_mix": {"rms_dbfs": second_db, "mix_transient_events_per_s": second_rate},
         "target": {"ins_pattern": "full"}}]}


def test_unmapped_blueprint_explicitly_reports_generic_control_provenance(tmp_path):
    files = source_files(tmp_path)
    source_bp = tmp_path / "unmapped.json"
    write_json(source_bp, measured_blueprint())
    manifest = prepare(*files, tmp_path / "out", source_bp, allow_long=True)
    mapping = manifest["arrangement_mapping"]
    assert manifest["workflow"] == "generic_ins_control"
    assert mapping["provenance"] == "generic_control"
    assert mapping["matched_plan_section_count"] == 0
    assert mapping["unmatched_plan_section_count"] == 2
    assert len(mapping["unused_blueprint_sections"]) == 2
    assert "not reference-driven" in mapping["warning"]


def test_measured_reference_proposals_really_change_arranged_audio_plan():
    excerpt, _ = plan.select_excerpt(ABC)
    rise = measured_blueprint(-24, -10, .4, 2)
    fall = measured_blueprint(-10, -24, 2, .4)
    rise_mapped = propose_mapping(rise)
    fall_mapped = propose_mapping(fall)
    rise_plan, rise_report = plan.edit_instrumental(excerpt, "arranged", rise_mapped)
    fall_plan, fall_report = plan.edit_instrumental(excerpt, "arranged", fall_mapped)
    assert rise_plan != fall_plan
    assert [c["mode"] for c in rise_report["changes"]] == ["sparse", "full"]
    assert [c["mode"] for c in fall_report["changes"]] == ["full", "sparse"]
    assert rise_report["mapping"]["provenance"] == "source_blueprint"
    assert rise_report["mapping"]["matched_plan_section_count"] == 2
    assert [s["name"] for s in rise_mapped["sections"]] == ["section_07", "section_08"]
    assert [s["name"] for s in fall_mapped["sections"]] == ["section_07", "section_08"]
    assert rise_mapped["mapping_proposal"]["status"] == "provisional_source_role_unknown"
    assert not rise_mapped["mapping_proposal"]["human_listening_reviewed"]
    assert not any(s["target"].get("plan_section") for s in rise["sections"])
    assert plan.vocal_identity(*plan.parse(rise_plan)) == plan.vocal_identity(*plan.parse(fall_plan))


def test_no_measurable_reference_contrast_proposes_no_fake_mapping():
    mapped = propose_mapping(measured_blueprint(-20, -20, 1, 1))
    assert mapped["mapping_proposal"]["status"] == "no_measurable_contrast_no_mapping_proposed"
    assert not any(s["target"].get("plan_section") for s in mapped["sections"])


def test_proposal_keeps_existing_explicit_mapping_unless_replacement_is_requested():
    mapped = propose_mapping(measured_blueprint())
    with pytest.raises(ValueError, match="already has explicit"):
        propose_mapping(mapped)
    updated = propose_mapping(mapped, target_sections=["bridge", "chorus"], replace=True)
    assert updated["sections"][0]["target"]["plan_section"] == "bridge"


def test_short_excerpt_requires_explicit_override(tmp_path):
    with pytest.raises(ValueError, match="30-60"):
        prepare(*source_files(tmp_path), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_rejects_unknown_abc_or_wrong_bar_duration():
    with pytest.raises(ValueError, match="Unsupported"):
        plan.bars(["[CEG]16|"], Fraction(16))
    with pytest.raises(ValueError, match="duration"):
        plan.bars(["C4D4|"], Fraction(16))
    with pytest.raises(ValueError, match="Unsupported chord"):
        plan.edit_instrumental(ABC.replace('"Am"', '"Asus4"'), "full")


def test_source_ins_mismatch_is_reported_and_not_allowed_to_shift_the_vocal():
    source = ABC.replace('"G"d8z8|', '"G"d8z8|').replace("Z4|\n% chorus", "Z5|\n% chorus")
    excerpt, info = plan.select_excerpt(source)
    assert info["warnings"]
    edited, changes = plan.edit_instrumental(excerpt, "full")
    assert changes["changes"][0]["source_ins_bars"] == 5
    assert changes["changes"][0]["bars"] == 4


def test_lyrics_keep_prechorus_and_correct_repeated_occurrence():
    first = plan.select_lyrics(LYRICS, ["% verse", "% chorus"])
    assert "Build to the chorus" in first and "Second verse" not in first
    second = plan.select_lyrics(LYRICS, ["% verse", "% chorus"], first_occurrence=2)
    assert "Second verse words" in second and "Second build" in second
    assert "First verse words" not in second


def test_plan_only_preparation_and_dry_run_never_start_subprocess(tmp_path, monkeypatch):
    out, manifest = make_experiment(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("dry run attempted subprocess"))
    result = dispatch(out, variants=["baseline", "full"], dry_run=True)
    assert result["status"] == "dry_run_no_gpu_started"
    assert len(result["commands"]) == 2
    for row in result["commands"]:
        assert "-I" in row["argv"] and "--offload-ar" in row["argv"]
        assert row["argv"][row["argv"].index("--backend") + 1] == "torch-eager"
    assert not (out / "renders").exists()
    assert manifest["claim_status"] == "unvalidated_experiment_no_quality_claim"


def test_changed_inputs_cannot_be_resumed_under_old_identity(tmp_path):
    out, _ = make_experiment(tmp_path)
    (out / "plans" / "full.abc").write_text(ABC, encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared plan changed"):
        checked_manifest(out)


def test_resume_checks_real_audio_hash(tmp_path, monkeypatch):
    out, manifest = make_experiment(tmp_path)
    calls = []

    def fake_renderer(argv, **kwargs):
        calls.append(argv)
        directory = out / "renders" / "baseline"
        sf.write(directory / "audio.wav", np.sin(np.arange(8000)*.1)*.1, 8000)
        write_json(directory / "result.json", {"experiment_identity": manifest["experiment_identity"], "truncated": {"abc": False, "semantic": False}})
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_renderer)
    kwargs = dict(variants=["baseline"], python=Path(sys.executable))
    assert dispatch(out, **kwargs)["variants"]["baseline"]["status"] == "complete"
    assert dispatch(out, **kwargs)["variants"]["baseline"]["status"] == "skipped_verified_complete"
    assert len(calls) == 1
    with (out / "renders" / "baseline" / "audio.wav").open("ab") as f:
        f.write(b"changed")
    assert dispatch(out, **kwargs)["variants"]["baseline"]["status"] == "complete"
    assert len(calls) == 2


def test_truncated_generation_never_gets_a_complete_checkpoint(tmp_path, monkeypatch):
    out, manifest = make_experiment(tmp_path)

    def fake_renderer(argv, **kw):
        directory = out / "renders" / "full"
        sf.write(directory / "audio.wav", np.ones(800)*.1, 8000)
        write_json(directory / "result.json", {"experiment_identity": manifest["experiment_identity"], "truncated": {"semantic": True}})
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_renderer)
    result = dispatch(out, variants=["full"], python=Path(sys.executable))
    assert result["status"] == "failed"
    assert "truncated" in result["variants"]["full"]["error"]


def test_common_safe_level_removes_loudness_confounds_without_clipping(tmp_path):
    sr = 8000
    t = np.arange(sr) / sr
    normal = .2*np.sin(2*np.pi*440*t)
    crest = normal.copy()
    crest[0] = .99
    files = {"normal": tmp_path/"normal.wav", "quiet": tmp_path/"quiet.wav", "crest": tmp_path/"crest.wav"}
    for label, y in [("normal", normal), ("quiet", normal*.01), ("crest", crest)]:
        sf.write(files[label], y, sr, subtype="FLOAT")
    result = loudness_match(files, tmp_path/"listening", target_dbfs=-6)
    levels = []
    for label in files:
        y, _ = sf.read(tmp_path/"listening"/f"{label}.wav")
        levels.append(np.sqrt(np.mean(y**2)))
        assert np.max(np.abs(y)) <= .980001
        assert result["files"][label]["source_sha256"] == digest(files[label])
        assert result["files"][label]["listening_sha256"] == digest(tmp_path/"listening"/f"{label}.wav")
    assert max(levels) - min(levels) < 1e-6
    assert result["common_rms_dbfs"] < -6


def test_stereo_analysis_downmix_is_arithmetic_mean_without_equal_power_boost(tmp_path):
    sr = 8000
    t = np.arange(sr)/sr
    left = .4*np.sin(2*np.pi*440*t)
    right = .2*np.sin(2*np.pi*440*t)
    p = tmp_path/"correlated_stereo.wav"
    sf.write(p, np.column_stack([left, right]), sr, subtype="FLOAT")
    analyzed = decode(p, sr)
    assert np.max(np.abs(analyzed - (left+right)/2)) < 1e-7
    native = native_file_levels(p)
    assert abs(native["sample_peak_dbfs"] - 20*np.log10(.4)) < .001
    assert abs(native["rms_dbfs"] - 20*np.log10(.3/np.sqrt(2))) < .001


def test_report_hashes_match_source_and_listening_files_and_peak_is_native(tmp_path):
    out, _ = make_experiment(tmp_path)
    directory = out/"renders"/"baseline"
    directory.mkdir(parents=True)
    sr = 48000
    square = np.sign(np.sin(2*np.pi*440*np.arange(sr)/sr))
    original = directory/"audio.wav"
    sf.write(original, np.column_stack([square, square]), sr, subtype="FLOAT")
    write_json(directory/"state.json", {"status": "complete", "audio_sha256": digest(original)})
    result = report(out)
    levels = result["loudness_match"]["files"]["baseline"]
    assert levels["source_sha256"] == digest(original)
    assert levels["listening_sha256"] == digest(Path(levels["listening_file"]))
    whole = result["metrics"]["baseline"]["whole_mix"]
    assert whole["sample_peak_dbfs"] == 0.0
    assert "no_resampling" in whole["level_measurement_basis"]
    assert abs(whole["rms_dbfs"] - native_file_levels(original)["rms_dbfs"]) < 1e-9


def test_silent_audio_cannot_be_normalized_or_mistaken_for_drums(tmp_path):
    zeros = np.zeros(8000, dtype=np.float32)
    values = descriptors(zeros, 8000)
    assert values["beat_estimate"] is None
    assert values["mix_transient_events_per_s"] == 0
    assert values["instrument_identity"] == "unknown_from_mix"
    p = tmp_path/"silent.wav"
    sf.write(p, zeros, 8000)
    with pytest.raises(ValueError, match="Silent audio"):
        loudness_match({"silent": p}, tmp_path/"listening")


def test_annotations_cannot_overlap_or_exceed_reference():
    data = {"sections": [{"name": "verse", "start_s": 0, "end_s": 12}, {"name": "chorus", "start_s": 11, "end_s": 20}]}
    with pytest.raises(ValueError, match="overlap"):
        validate_annotations(data, 30)
    data["sections"][1]["start_s"] = 12
    with pytest.raises(ValueError, match="exceed"):
        validate_annotations(data, 19)


def test_blueprint_records_uncertainty_and_stems_must_align(tmp_path):
    sr = 8000
    source = tmp_path/"reference.wav"
    sf.write(source, np.sin(np.arange(sr*2)*.2)*.1, sr)
    annotations = tmp_path/"annotations.json"
    write_json(annotations, {"status": "draft", "tempo_bpm": None,
                             "sections": [{"name": "verse", "start_s": 0, "end_s": 1}, {"name": "chorus", "start_s": 1, "end_s": 2}]})
    bp = blueprint(source, annotations)
    assert bp["annotation_status"] == "draft"
    assert bp["sections"][0]["role_observation_status"] == "unknown_no_instrument_detector"
    assert bp["sections"][0]["target"]["proposal_status"] == "hypothesis_requires_listening"
    short = tmp_path/"stem.wav"
    sf.write(short, np.zeros(sr), sr)
    with pytest.raises(ValueError, match="not time-aligned"):
        blueprint(source, annotations, {"drums": str(short)})


def test_empty_report_is_actionable_and_never_awards_a_score(tmp_path):
    out, _ = make_experiment(tmp_path)
    result = report(out)
    assert result["status"] == "awaiting_render"
    assert result["promotion_status"] == "manual_decision_required"
    assert result["human_ratings"]["preferred_variant"] is None
    assert (out/"REPORT.md").exists()


def reusable_experiments(tmp_path, monkeypatch, variant="baseline", cot="full", cfg_scale=None, semantic_max_tokens=None):
    files = source_files(tmp_path)
    source, target = tmp_path/"original", tmp_path/"mapped"
    source_manifest = prepare(*files, source, allow_long=True, cot=cot, cfg_scale=cfg_scale, semantic_max_tokens=semantic_max_tokens)
    bp_file = tmp_path/"mapped.json"
    write_json(bp_file, propose_mapping(measured_blueprint()))
    target_manifest = prepare(*files, target, bp_file, allow_long=True, cot=cot, cfg_scale=cfg_scale, semantic_max_tokens=semantic_max_tokens)
    runtime = {"runtime_sha256": "verified-installed-runtime", "weights": {"mot": "verified-model", "vae": "verified-vae"}}
    monkeypatch.setattr(reuse_module, "_runtime_model_identity", lambda python: runtime)
    directory = source/"renders"/variant
    artifacts = directory/"artifacts"
    artifacts.mkdir(parents=True)
    sf.write(directory/"audio.wav", np.sin(np.arange(8000)*.1)*.1, 8000)
    command = render_commands(source, [variant], python=Path(sys.executable))[0][1]
    write_json(directory/"state.json", {"status": "complete", "request_identity": "original-request",
                                        "audio_sha256": digest(directory/"audio.wav"), "command": command})
    write_json(directory/"result.json", {"experiment_identity": source_manifest["experiment_identity"],
                                         "variant": variant, "seed": 1338,
                                         "rendering": rendering_config(source_manifest),
                                         "truncated": {"abc": False, "semantic": False}})
    request = {"style": (source/"style.txt").read_text(), "lyrics": (source/"lyrics.txt").read_text(),
               "abc": (source/"plans"/f"{variant}.abc").read_text(), "seed": 1338, "cot": cot, "cfg_scale": cfg_scale}
    semantic = dict(SEMANTIC_DEFAULTS)
    overrides = {}
    if semantic_max_tokens is not None:
        semantic["max_tokens"] = semantic_max_tokens
        if semantic != SEMANTIC_DEFAULTS:
            overrides["semantic"] = semantic
    if cfg_scale is not None and cfg_scale != 1.0:
        overrides["cfg_scale"] = cfg_scale
    config = {"backend": "torch-eager", "memory_budget_gib": 12, "offload_ar": True, "cot": cot,
              "cfg_scale": cfg_scale if cfg_scale is not None else 1.0, "quantization": "none",
              "generation": {"semantic": semantic}, "overrides": overrides, "runtime_sha256": runtime["runtime_sha256"]}
    write_json(artifacts/"request.json", request)
    write_json(artifacts/"config.json", config)
    (artifacts/"score.abc").write_text(request["abc"])
    write_json(artifacts/"result.json", {"status": "complete", "identity": "native-original-request",
                                         "truncated": {"abc": False, "semantic": False}, "weights": runtime["weights"],
                                         "artifacts": {p.name: {"sha256": digest(p)} for p in artifacts.iterdir()}})
    return source, target, source_manifest, target_manifest


def test_explicit_input_equivalent_reuse_records_origin_and_performs_no_inference(tmp_path, monkeypatch):
    source, target, original, mapped = reusable_experiments(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("reuse attempted inference or a subprocess"))
    result = reuse_renders(source, target, ["baseline"], python=Path(sys.executable))
    assert result["execution"] == "explicit_verified_reuse_no_new_inference"
    directory = target/"renders"/"baseline"
    state = json.loads((directory/"state.json").read_text())
    copied_result = json.loads((directory/"result.json").read_text())
    assert state["status"] == "complete" and state["execution"] == "input_equivalent_reuse"
    assert not state["inference_performed"] and not copied_result["inference_performed"]
    assert copied_result["experiment_identity"] == mapped["experiment_identity"]
    assert state["reuse_lineage"]["source_experiment_identity"] == original["experiment_identity"]
    assert digest(directory/"audio.wav") == digest(source/"renders"/"baseline"/"audio.wav")
    assert (directory/"reuse_source"/"artifacts"/"request.json").exists()
    again = reuse_renders(source, target, ["baseline"], python=Path(sys.executable))
    assert again["variants"]["baseline"]["status"] == "skipped_verified_input_equivalent_reuse"
    # Dispatch recognizes the complete equivalent request without another run.
    resumed = dispatch(target, variants=["baseline"], python=Path(sys.executable))
    assert resumed["variants"]["baseline"]["status"] == "skipped_verified_complete"


@pytest.mark.parametrize("mismatch", ["plan", "seed", "runtime", "incomplete", "audio_hash", "truncated", "native_request", "model"])
def test_reuse_refuses_mismatched_or_unverified_sources_before_destination_writes(tmp_path, monkeypatch, mismatch):
    source, target, _, target_manifest = reusable_experiments(tmp_path, monkeypatch)
    directory = source/"renders"/"baseline"
    if mismatch == "plan":
        p = target/"plans"/"baseline.abc"
        p.write_text(p.read_text().replace('"Am"c4', '"Am"d4'))
        target_manifest["variants"]["baseline"]["plan_sha256"] = digest(p)
        write_json(target/"manifest.json", target_manifest)
    elif mismatch == "seed":
        target_manifest["variants"]["baseline"]["seed"] = 999
        write_json(target/"manifest.json", target_manifest)
    elif mismatch in {"runtime", "incomplete", "audio_hash"}:
        state = json.loads((directory/"state.json").read_text())
        if mismatch == "runtime":
            state["command"][state["command"].index("--nar-query-chunk")+1] = "2048"
        elif mismatch == "incomplete":
            state["status"] = "running"
        else:
            state["audio_sha256"] = "bad-audio-hash"
        write_json(directory/"state.json", state)
    elif mismatch == "truncated":
        result = json.loads((directory/"result.json").read_text())
        result["truncated"]["semantic"] = True
        write_json(directory/"result.json", result)
    elif mismatch == "native_request":
        p = directory/"artifacts"/"request.json"
        request = json.loads(p.read_text())
        request["lyrics"] = "Actually sang different words"
        write_json(p, request)
        native_file = directory/"artifacts"/"result.json"
        native = json.loads(native_file.read_text())
        native["artifacts"]["request.json"]["sha256"] = digest(p)
        write_json(native_file, native)
    else:
        monkeypatch.setattr(reuse_module, "_runtime_model_identity", lambda python: {"runtime_sha256": "verified-installed-runtime", "weights": {"mot": "changed"}})
    with pytest.raises(ValueError):
        reuse_renders(source, target, ["baseline"], python=Path(sys.executable))
    assert not (target/"renders").exists()


@pytest.mark.parametrize("setting", [{"cot": "melody", "cfg_scale": None}, {"cot": "full", "cfg_scale": 1.5},
                                     {"cot": "full", "cfg_scale": None, "semantic_max_tokens": 1200}])
def test_reuse_refuses_different_cot_or_cfg_even_if_input_files_match(tmp_path, monkeypatch, setting):
    source, target, _, target_manifest = reusable_experiments(tmp_path, monkeypatch)
    target_manifest["rendering"] = setting
    write_json(target/"manifest.json", target_manifest)
    with pytest.raises(ValueError, match="cot/cfg_scale"):
        reuse_renders(source, target, ["baseline"], python=Path(sys.executable))
    assert not (target/"renders").exists()


def test_reuse_accepts_matching_melody_cfg_and_semantic_ceiling_but_verifies_native_max(tmp_path, monkeypatch):
    source, target, _, _ = reusable_experiments(tmp_path, monkeypatch, cot="melody", cfg_scale=1.5, semantic_max_tokens=1200)
    native_config = source/"renders"/"baseline"/"artifacts"/"config.json"
    original_config = json.loads(native_config.read_text())
    result = reuse_renders(source, target, ["baseline"], python=Path(sys.executable))
    assert result["variants"]["baseline"]["status"] == "input_equivalent_reuse"
    # A valid source config hash alone is insufficient if the recorded sampling
    # ceiling differs from the prepared request.
    original_config["generation"]["semantic"]["max_tokens"] = 2400
    write_json(native_config, original_config)
    native_result_file = native_config.with_name("result.json")
    native = json.loads(native_result_file.read_text())
    native["artifacts"]["config.json"]["sha256"] = digest(native_config)
    write_json(native_result_file, native)
    with pytest.raises(ValueError, match="runtime configuration"):
        reuse_renders(source, target, ["baseline"], python=Path(sys.executable))


def test_reuse_report_explicitly_marks_no_new_independent_inference(tmp_path, monkeypatch):
    source, target, _, _ = reusable_experiments(tmp_path, monkeypatch)
    reuse_renders(source, target, ["baseline"], python=Path(sys.executable))
    result = report(target)
    assert result["render_executions"]["baseline"] == "input_equivalent_reuse"
    assert result["reuse_lineage"]["baseline"]["inference_performed_for_destination"] is False
    assert "not an independent replication" in (target/"REPORT.md").read_text()
    assert result["status"] == "technical_review_required"
    assert result["metrics"]["baseline"]["duration_validation"]["technical_review_required"]
    assert all(section["values"] is None for section in result["metrics"]["baseline"]["sections"])
    assert "baseline" in result["technical_review_variants"]


def test_local_iron_rain_complete_verse_chorus_when_available():
    source = ROOT/"_archive/outputs_history/2026-10-16_epic_hook/candidates/04_iron_rain_20261016-150430/04_iron_rain_c2_artifacts/score.abc"
    if not source.exists():
        pytest.skip("Local generated take not shipped with repository")
    excerpt, info = plan.select_excerpt(source.read_text(encoding="utf-8"))
    assert info["bars"] == 21 and 58 < info["duration_s"] < 59
    assert info["sections"] == ["% verse", "% chorus"]
    for mode in ("sparse", "full", "arranged"):
        edited, changes = plan.edit_instrumental(excerpt, mode)
        assert changes["vocal_header_unchanged"]
        assert plan.vocal_identity(*plan.parse(edited)) == info["vocal_identity"]
