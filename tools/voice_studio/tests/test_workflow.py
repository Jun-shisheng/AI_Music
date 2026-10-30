from pathlib import Path
import json
import sys

import numpy as np
import pytest

from tools.voice_studio import audio, workflow


def test_context_merge_preserves_entire_timeline_and_samples():
    rate=audio.RATE
    # Force many cuts across voice and silence, and a short final chunk.
    signal=np.sin(np.arange(rate*31+317)*.031).astype(np.float32)*.4
    signal[rate*9:rate*10]=0
    chunks=audio.segments(signal,rate,max_seconds=7,context_seconds=.2)
    result=audio.merge([(c,signal[c["context_start"]:c["context_end"]]) for c in chunks],len(signal))
    assert len(chunks)>4
    np.testing.assert_allclose(signal,result,atol=1e-7)
    assert len(result)==len(signal)


def test_bad_chunk_length_is_not_silently_stretched():
    entry=dict(index=0,start=0,end=100,context_start=0,context_end=100)
    with pytest.raises(ValueError,match="duration mismatch"):
        audio.merge([(entry,np.zeros(99))],100)


def test_audio_quality_detects_missing_phrases():
    signal=np.ones(audio.RATE*5)*.12
    output=signal.copy()
    output[audio.RATE:audio.RATE*3]=0
    qa=audio.technical_qa(signal,output)
    assert qa["dropout_ratio"]>.3
    assert qa["warnings"]
    assert qa["identity_score"] is None
    assert qa["listening_required"]


def test_candidates_avoid_silent_intro_and_are_distinct():
    signal=np.zeros(audio.RATE*30)
    signal[audio.RATE*10:]=np.sin(np.arange(audio.RATE*20)*.1)*.2
    choices=audio.reference_candidates(signal,audio.RATE,seconds=6,count=3)
    assert choices[0]["start"]>=audio.RATE*10
    assert all(x["active_ratio"]>.9 for x in choices)
    assert abs(choices[0]["start"]-choices[1]["start"])>=audio.RATE*6


def test_score_rejects_unaligned_lyrics_and_duration_mismatch(tmp_path):
    path=tmp_path/"score.json"
    path.write_text(json.dumps([dict(time=[0,1000],phoneme="a b",duration=".1 .1",note_pitch="60 62",note_type="1 1")]))
    with pytest.raises(ValueError,match="durations"):
        workflow.validate_metadata(path)
    path.write_text(json.dumps([dict(time=[0,1000],lyrics="new lyrics")]))
    with pytest.raises(ValueError,match="phoneme"):
        workflow.validate_metadata(path)


def test_traversal_ids_rejected():
    for value in ["../oops","a/b","..","a\\b"]:
        with pytest.raises(ValueError):
            workflow.safe_id(value)


def prepared_profile(tmp_path):
    base=tmp_path/"profiles/test_voice"
    base.mkdir(parents=True)
    reference=base/"reference.wav"
    f0=base/"reference_f0.npy"
    audio.write(reference,np.ones(audio.RATE*3)*.1)
    np.save(f0,np.ones(150)*220)
    manifest=dict(voice_id="test_voice",status="pending_listening",preparation_status="reference_ready",
        artifacts={p.stem:dict(path=str(p),sha256=audio.digest(p)) for p in [reference,f0]})
    audio.atomic_json(base/"profile.json",manifest)
    return base/"profile.json"


def base_args(tmp_path,source,command="cover"):
    return workflow.parser().parse_args([command,"--workspace",str(tmp_path),"--voice-id","test_voice",
        "--source",str(source),"--seconds","0","--job-id","test_job",
        "--models",str(tmp_path/"models"),"--soulx-root",str(tmp_path/"source"),
        "--backend-python",sys.executable])


def test_relyric_cannot_relabel_old_audio_with_new_lyrics(tmp_path):
    prepared_profile(tmp_path)
    source=tmp_path/"source.wav"
    audio.write(source,np.ones(audio.RATE)*.1)
    args=base_args(tmp_path,source,"relyric")
    with pytest.raises(ValueError,match="guide-vocals"):
        workflow.conversion(args)


def test_missing_model_writes_failed_manifest_without_claiming_audio(tmp_path):
    prepared_profile(tmp_path)
    source=tmp_path/"source.wav"
    audio.write(source,np.ones(audio.RATE)*.1)
    args=base_args(tmp_path,source)
    args.models=tmp_path/"missing_models"
    args.execute=True
    with pytest.raises(FileNotFoundError,match="Backend"):
        workflow.conversion(args)
    result=json.loads((tmp_path/"jobs/test_job/manifest.json").read_text())
    assert result["status"]=="failed"
    assert result["outputs"]=={}
    assert result["listening_approved"] is False


def test_same_named_changed_audio_cannot_reuse_job(tmp_path):
    prepared_profile(tmp_path)
    source=tmp_path/"source.wav"
    audio.write(source,np.ones(audio.RATE)*.1)
    args=base_args(tmp_path,source)
    first=workflow.conversion(args)
    assert first["status"]=="planned"
    audio.write(source,np.ones(audio.RATE)*.2)
    with pytest.raises(ValueError,match="different inputs"):
        workflow.conversion(args)


def test_modified_reference_fails_instead_of_silently_changing_identity(tmp_path):
    profile_path=prepared_profile(tmp_path)
    source=tmp_path/"source.wav"
    audio.write(source,np.ones(audio.RATE)*.1)
    audio.write(profile_path.parent/"reference.wav",np.ones(audio.RATE*3)*.2)
    with pytest.raises(ValueError,match="profile has changed"):
        workflow.conversion(base_args(tmp_path,source))


def test_pipeline_exports_exact_dry_timeline_and_human_review_state(tmp_path,monkeypatch):
    prepared_profile(tmp_path)
    source=tmp_path/"source.wav"
    signal=np.sin(np.arange(audio.RATE*14+93)*.04)*.2
    audio.write(source,signal)
    args=base_args(tmp_path,source)
    args.vocals=source
    args.execute=True
    monkeypatch.setattr(workflow,"model_spec",lambda *unused:dict(model_sha256="fake-test-model"))
    def fake_f0(unused,files,directory):
        for _,out in files:
            np.save(out,np.ones(50)*220)
    def fake_worker(unused,task,spec,directory):
        assert task=="svc"
        for chunk in spec["chunks"]:
            # An identity converter exercises timeline/crossfade/QA without a GPU.
            data,rate=audio.read(Path(chunk["audio"]))
            audio.write(Path(chunk["output"]),data,rate)
    monkeypatch.setattr(workflow,"f0",fake_f0)
    monkeypatch.setattr(workflow,"worker",fake_worker)
    result=workflow.conversion(args)
    assert result["status"]=="awaiting_listening"
    assert result["listening_approved"] is False
    output,_=audio.read(Path(result["outputs"]["vocals"]["path"]))
    expected,_=audio.read(source)
    np.testing.assert_allclose(output,expected,atol=1/32768)
    assert result["technical_qa"]["identity_score"] is None
    assert "mix" not in result["outputs"]


def test_f0_cache_is_invalidated_when_reference_audio_changes(tmp_path,monkeypatch):
    source=tmp_path/"reference.wav"
    output=tmp_path/"reference_f0.npy"
    calls=[]
    def extract(args,task,spec,directory,**kwargs):
        calls.append(spec)
        np.save(output,np.ones(30)*220)
    monkeypatch.setattr(workflow,"worker",extract)
    audio.write(source,np.ones(audio.RATE)*.1)
    workflow.f0(None,[(source,output)],tmp_path)
    workflow.f0(None,[(source,output)],tmp_path)
    assert len(calls)==1
    audio.write(source,np.ones(audio.RATE)*.2)
    workflow.f0(None,[(source,output)],tmp_path)
    assert len(calls)==2


def test_prompt_metadata_must_start_at_zero(tmp_path):
    metadata=tmp_path/"prompt.json"
    metadata.write_text(json.dumps([dict(time=[1000,7000],phoneme="a",duration="6",note_pitch="60",note_type="1")]))
    with pytest.raises(ValueError,match="0ms"):
        workflow.validate_metadata(metadata,prompt=True)


def test_asset_verification_rejects_same_length_corruption(tmp_path):
    import hashlib
    from tools.voice_studio.download_models import verify
    path=tmp_path/"weight.pt"
    path.write_bytes(b"good")
    expected=dict(size=4,sha256=hashlib.sha256(b"good").hexdigest())
    assert verify(path,expected)["verified"]
    path.write_bytes(b"evil")
    with pytest.raises(ValueError,match="SHA-256"):
        verify(path,expected)


def test_level_match_is_one_gain_with_quiet_guard_and_gain_bound():
    n=audio.RATE*3+157
    source=np.sin(np.arange(n)*.043)*.15
    source[:audio.RATE]=0
    raw=source*.5
    matched,report=audio.match_vocal_level(source,raw)
    assert len(matched)==n
    assert report["time_varying_gain"] is False
    np.testing.assert_allclose(matched,raw*report["linear_gain"],atol=1e-8)
    assert abs(report["linear_gain"]-2)<.005
    assert np.count_nonzero(matched[:audio.RATE])==0
    _,quiet_report=audio.match_vocal_level(source,raw*.001)
    assert quiet_report["reason"]=="generated_vocal_too_quiet_to_boost"
    assert quiet_report["linear_gain"]==1
    _,bounded=audio.match_vocal_level(source,raw*.1)
    assert bounded["gain_db"]==6
    assert bounded["gain_limited"]


def test_static_mix_preserves_dynamics_with_one_global_gain():
    n=audio.RATE*2+107
    t=np.arange(n)
    voice=np.sin(t*.033)*.7
    background=np.stack([np.sin(t*.045)*.8,np.sin(t*.025)*.6],axis=1)
    mixed,report=audio.static_mix(background,voice,n,3)
    expected=background+voice[:,None]*10**(3/20)
    assert mixed.shape==(n,2)
    assert report["global_linear_gain"]<1
    assert report["time_varying_gain"] is False
    assert np.max(np.abs(mixed))<=.950001
    np.testing.assert_allclose(mixed,expected*report["global_linear_gain"],atol=1e-7)


def test_real_wav_mix_retains_duration_and_constant_gain_without_clipping(tmp_path):
    n=audio.RATE*2+123
    voice=np.sin(np.arange(n)*.033)*.7
    background=np.stack([np.sin(np.arange(n)*.045)*.8]*2,axis=1)
    instrumental=tmp_path/"instrumental.wav"
    vocals=tmp_path/"vocals.wav"
    output=tmp_path/"mix.wav"
    audio.write(instrumental,background)
    audio.write(vocals,voice)
    source_hashes=audio.digest(instrumental),audio.digest(vocals)
    report=workflow.remix(instrumental,vocals,output,n,0)
    actual,rate=audio.read(output)
    bg,_=audio.read(instrumental)
    v,_=audio.read(vocals)
    expected=(bg+v)*report["global_linear_gain"]
    assert len(actual)==n and rate==audio.RATE
    assert np.max(np.abs(actual))<.951
    np.testing.assert_allclose(actual,expected,atol=1/32768)
    assert (audio.digest(instrumental),audio.digest(vocals))==source_hashes


def test_doctor_missing_necessary_assets_returns_failure_exit(tmp_path,capsys):
    status=workflow.main(["doctor","--models",str(tmp_path/"missing"),"--workspace",str(tmp_path)])
    assert status!=0
    report=json.loads(capsys.readouterr().out)
    assert report["status"]=="dependencies_missing"
    assert report["paths"]["whisper_weights"]["exists"] is False


@pytest.mark.parametrize("asset",["model.safetensors","config.json","preprocessor_config.json"])
def test_each_whisper_asset_change_invalidates_conversion_cache(tmp_path,asset):
    prepared_profile(tmp_path)
    source=tmp_path/"source.wav"
    audio.write(source,np.ones(audio.RATE)*.1)
    args=base_args(tmp_path,source)
    args.job_id=None
    whisper=args.models/"whisper-base"
    whisper.mkdir(parents=True)
    (whisper/asset).write_bytes(b"first-content")
    before=workflow.conversion(args)
    (whisper/asset).write_bytes(b"other-content")
    after=workflow.conversion(args)
    assert before["cache_key"]!=after["cache_key"]
    assert before["manifest"]!=after["manifest"]
    assert before["inputs"]["whisper_assets"][asset]["sha256"]!=after["inputs"]["whisper_assets"][asset]["sha256"]


@pytest.mark.parametrize("package",["torch","torchaudio","transformers","tokenizers","numpy"])
def test_backend_package_version_change_invalidates_conversion_cache(tmp_path,monkeypatch,package):
    prepared_profile(tmp_path)
    source=tmp_path/"source.wav"
    audio.write(source,np.ones(audio.RATE)*.1)
    args=base_args(tmp_path,source)
    args.job_id=None
    versions={name:"version-a" for name in ["torch","torchaudio","transformers","tokenizers","numpy"]}
    monkeypatch.setattr(workflow,"backend_identity",lambda unused:dict(packages=versions.copy(),status="metadata_read"))
    before=workflow.conversion(args)
    versions[package]="version-b"
    after=workflow.conversion(args)
    assert before["cache_key"]!=after["cache_key"]
    assert before["inputs"]["backend_runtime"]["packages"][package]=="version-a"
    assert after["inputs"]["backend_runtime"]["packages"][package]=="version-b"


def svs_args(tmp_path):
    profile=prepared_profile(tmp_path)
    source=tmp_path/"source.wav"
    audio.write(source,np.ones(audio.RATE)*.1)
    target=tmp_path/"external_target.json"
    metadata=[dict(time=[0,1000],duration="1",phoneme="en_N-UW1",text="new",note_pitch="60",note_type="2")]
    audio.atomic_json(target,metadata)
    lyrics=tmp_path/"lyrics.txt";lyrics.write_text("new")
    prompt=profile.parent/"prompt_metadata.json"
    audio.atomic_json(prompt,[dict(time=[0,3000],duration="3",phoneme="en_AA1",text="ah",note_pitch="60",note_type="2")])
    singer=workflow.load(profile)
    singer["artifacts"]["prompt_metadata"]=dict(path=str(prompt),sha256=audio.digest(prompt))
    audio.atomic_json(profile,singer)
    args=base_args(tmp_path,source,"relyric")
    args.mode="svs";args.score_json=target;args.lyrics_file=lyrics
    return args


def test_svs_score_and_melody_controls_cannot_share_cache(tmp_path):
    args=svs_args(tmp_path);args.job_id=None
    args.control="score";score=workflow.conversion(args)
    args.control="melody";melody=workflow.conversion(args)
    assert score["cache_key"]!=melody["cache_key"]
    assert score["inputs"]["control"]=="score"
    assert melody["inputs"]["control"]=="melody"


@pytest.mark.parametrize("external_change",["delete","modify"])
def test_svs_worker_consumes_verified_archive_after_external_score_changes(tmp_path,monkeypatch,external_change):
    args=svs_args(tmp_path);args.execute=True
    original_score=args.score_json.read_bytes()
    expected_hash=audio.digest(args.score_json)
    monkeypatch.setattr(workflow,"model_spec",lambda *unused:dict(model_sha256="fake-test-model"))
    def fake_worker(unused,task,spec,directory):
        assert task=="svs"
        archive=Path(spec["target_metadata"])
        assert archive==directory/"target_metadata.json"
        assert archive!=args.score_json
        if external_change=="delete":
            args.score_json.unlink()
        else:
            args.score_json.write_text("changed outside job")
        assert archive.read_bytes()==original_score
        assert audio.digest(archive)==expected_hash
        audio.write(Path(spec["save_dir"])/"generated.wav",np.ones(audio.RATE)*.1)
    monkeypatch.setattr(workflow,"worker",fake_worker)
    result=workflow.conversion(args)
    archived=Path(result["archived_target_metadata"]["path"])
    assert archived.read_bytes()==original_score
    assert audio.digest(archived)==result["inputs"]["score_sha256"]==expected_hash
    assert result["status"]=="awaiting_listening"


def test_score_archive_refuses_mismatched_input_or_existing_copy(tmp_path):
    source=tmp_path/"source.json";source.write_text("original")
    expected=audio.digest(source)
    directory=tmp_path/"job";directory.mkdir()
    archive=workflow.archive_score(source,directory,expected)
    archive.write_text("tampered archive")
    with pytest.raises(ValueError,match="Archived target metadata"):
        workflow.archive_score(source,directory,expected)
    source.write_text("changed source")
    with pytest.raises(ValueError,match="changed after task identity"):
        workflow.archive_score(source,directory,expected)
