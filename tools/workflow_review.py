"""Package real audio outputs into an offline A/B listening sheet."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DELIVERY = ROOT / "_archive/outputs_history/2026-10-19_quality_upgrade"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_artifact(record: dict) -> Path:
    path = Path(record.get("path", "__missing__"))
    expected = record.get("sha256")
    return path if expected and path.is_file() and sha(path) == expected else Path("__missing__")


def ready_render(experiment: Path, variant: str) -> Path:
    state = load(experiment / f"renders/{variant}/state.json")
    path = experiment / f"renders/{variant}/audio.wav"
    if state.get("status") == "complete" and path.is_file() and sha(path) == state.get("audio_sha256"):
        return path
    return Path("__missing__")


def voice_output(manifest: dict) -> Path:
    if manifest.get("status") not in ("awaiting_listening", "technical_review_required"):
        return Path("__missing__")
    outputs = manifest.get("outputs", {})
    for key in ("mix", "vocals"):
        path = verified_artifact(outputs.get(key, {}))
        if path.is_file():
            return path
    return Path("__missing__")


def build(delivery: Path, workspace: Path) -> Path:
    destination = delivery / "listening"
    media = destination / "media"
    media.mkdir(parents=True, exist_ok=True)
    records, cards = [], []

    def audio(label: str, source: Path, key: str) -> str:
        if not source.is_file():
            return f'<p class="pending">{html.escape(label)}：等待生成</p>'
        target = media / f"{key}{source.suffix.lower()}"
        digest = sha(source)
        if not target.is_file() or sha(target) != digest:
            shutil.copy2(source, target)
        records.append({"label": label, "path": str(target.resolve()), "source": str(source.resolve()), "sha256": digest})
        return f'<label>{html.escape(label)}</label><audio controls preload="none" src="media/{html.escape(target.name)}"></audio>'

    profile = workspace / "profiles/iron_rain_female_v1/profile.json"
    profile_data = load(profile)
    anchor = verified_artifact(profile_data.get("artifacts", {}).get("reference", {}))
    cards.append('<section><h2>固定女声母本 · Iron Rain</h2><p>选取第4首 4–11 秒，先熟悉声音身份。档案仍等待听审。</p>' + audio("参考干声", anchor, "00_voice_anchor") + '</section>')
    for index, (job, title, source_name) in enumerate([
        ("cover_no_white_flag_v1", "跨歌曲翻唱 · No White Flag", "02_no_white_flag_35_55_source.wav"),
        ("cover_hammer_v1", "跨歌曲翻唱 · Hammer and Hymn", "01_hammer_and_hymn_35_55_source.wav"),
    ], 1):
        manifest = load(workspace / "jobs" / job / "manifest.json")
        mixed = voice_output(manifest)
        content = audio("原演唱与伴奏", delivery / "voice/preparation" / source_name, f"{index:02d}_original")
        content += audio("固定女声版本", mixed, f"{index:02d}_fixed_voice")
        qa_note = ("前后自动转写的词句保留较稳定，优先试听。" if job == "cover_no_white_flag_v1" else "前后自动转写差异较大，词句保留需要重点复核。")
        cards.append(f'<section data-item="{job}"><h2>{title}</h2><p>{qa_note}音色身份与自然度仍待听审；检查高音、咬字、接缝和节奏。</p>{content}{score_fields(job, ["音色相似", "自然度", "歌词清晰", "整体偏好"])}</section>')

    support = delivery / "reference/iron_rain_support"
    support_complete = all(ready_render(support, variant).is_file() for variant in ("baseline", "arranged"))
    experiment = support if support_complete else delivery / "reference/iron_rain_control"
    experiment_manifest = load(experiment / "manifest.json")
    report = load(experiment / "report.json")
    variants = ("baseline", "arranged")
    content = ''
    for variant, label in zip(variants, ("原计划基准", "主歌稀疏／副歌展开")):
        native = ready_render(experiment, variant)
        selected = native
        matched = (report.get("loudness_match") or {}).get("files", {}).get(variant, {})
        normalized = Path(matched.get("listening_file", "__missing__"))
        if (native.is_file() and report.get("experiment_identity") == experiment_manifest.get("experiment_identity")
                and matched.get("source_sha256") == sha(native) and normalized.is_file()
                and matched.get("listening_sha256") == sha(normalized)):
            selected = normalized
        if selected == native:
            label += "（原始响度）"
        content += audio(label, selected, f"reference_{variant}")
    reference_note = ("短风格描述＋温和根音／五度伴奏。两版时长稳定；编曲版目标词近似恢复9/9，仍有额外转写，当前为待审实验。" if support_complete else "首轮诊断样本：密集版时长偏离计划39%，歌词检查未通过；不作为质量提升成品。")
    cards.append('<section data-item="reference"><h2>编曲对照 · 完整主歌进入副歌</h2><p>' + reference_note + '相同人声计划、和弦、歌词与种子，只改变 Ins。参考映射是未听审提案。</p>' + content + score_fields("reference", ["段落对比", "律动", "层次清晰", "人声自然", "整体偏好"]) + '</section>')

    guide = delivery / "voice/new_lyrics_guide/renders/baseline/audio.wav"
    lyric_jobs = [workspace / f"jobs/{name}/manifest.json" for name in ("relyric_chorus_svs_v2", "relyric_chorus_svs_v1", "relyric_chorus_v1")]
    lyric_manifest = next((data for path in lyric_jobs if voice_output(data := load(path)).is_file()), {})
    result = voice_output(lyric_manifest)
    content = audio("新歌词＋固定女声（检查任务报告后试听）", result, "new_lyrics_fixed_voice")
    previous = voice_output(load(workspace / "jobs/relyric_chorus_svs_v1/manifest.json"))
    if result.is_file() and previous.is_file() and result != previous:
        content += '<details><summary>首版SVS对照</summary><p>新版只修正sparks的起音边界，保留首版用于比较。两版逐字准确性均须结合ASR与听审。</p>' + audio("首版SVS固定女声", previous, "new_lyrics_svs_v1") + '</details>'
    content += '<details><summary>首轮失败诊断</summary><p>这段导唱没有正确唱出目标新词，因此没有将其转声后当作填词成品。</p>' + audio("首轮导唱（目标新词检查未通过）", guide, "new_lyrics_guide") + '</details>'
    lyric_path = ROOT / "inputs/workflows/iron_rain_new_chorus.txt"
    lyrics = html.escape(lyric_path.read_text(encoding="utf-8")) if lyric_path.is_file() else ''
    lyric_note = ("直接SVS样本使用近似音素对齐；旋律整体降一个八度适配母本音区，调性与旋律轮廓保留。ASR近似恢复4/4句，前两句仍有错词／插词疑点，逐字准确性未通过验收。单独干声尚未做伴奏重编。" if result.is_file() and "svs" in str(result.parent) else "改词任务的完成状态以实际音频与任务报告为准。")
    cards.append('<section data-item="relyric"><h2>改词重唱 · 新副歌</h2><p>' + lyric_note + '</p><pre>' + lyrics + '</pre>' + content + score_fields("relyric", ["音色相似", "新词准确", "自然度", "整体偏好"]) + '</section>')

    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    payload = {"created_at": timestamp, "listening_status": "pending_user_listening", "audio": records,
               "automatic_quality_score": None, "profile": str(profile.resolve()),
               "reference_experiment": str(experiment.resolve()),
               "reference_experiment_identity": experiment_manifest.get("experiment_identity"),
               "relyric_job_manifest": lyric_manifest.get("manifest"),
               "relyric_cache_key": lyric_manifest.get("cache_key"),
               "user_context": "用户当前没有耳机，授权继续使用该参考；未完成主观验收。"}
    (destination / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Music · 两条工作流试听</title>
<style>body{margin:0;background:#111722;color:#e8edf5;font:16px/1.6 system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:35px 22px}h1{font-size:30px}h2{font-size:21px;margin-top:0}p{color:#b6c1d1}section{margin:20px 0;padding:24px;border:1px solid #344256;border-radius:16px;background:#1a2332}audio{display:block;width:100%;margin:8px 0 18px}label{color:#dbe7f8}.pending{color:#e1b778}.scores{display:flex;flex-wrap:wrap;gap:12px}.scores label{font-size:14px}.scores input{display:block;width:75px;padding:7px;margin-top:5px;background:#101824;color:#fff;border:1px solid #52627a;border-radius:6px}textarea{box-sizing:border-box;width:100%;padding:12px;background:#101824;color:#fff;border:1px solid #52627a;border-radius:8px}button{background:#91ccbc;color:#10271e;border:0;border-radius:9px;padding:12px 18px;font-size:16px;cursor:pointer}pre{white-space:pre-wrap;color:#c9d8ed}.small{font-size:13px}</style>
<main><h1>AI Music · 固定歌手与编曲对照</h1><p>先听母本，再比较跨曲音色、编曲和改词。评分留空表示尚未听过；没有自动生成的审美高分。</p><p class="small">生成：''' + timestamp + '''</p>''' + ''.join(cards) + '''<section><h2>听感记录</h2><textarea id="notes" rows="4" placeholder="例如：哪一版更好、具体哪几秒、哪里仍有机器感"></textarea><p><button id="save">导出试听评分 JSON</button></p><p class="small">评分只保存在当前页面和你导出的文件中，不会自动上传或替你验收。</p></section></main>
<script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));document.querySelector('#save').onclick=()=>{const invalid=[...document.querySelectorAll('input[data-item]')].find(i=>i.value!==''&&!i.checkValidity());if(invalid){invalid.reportValidity();invalid.focus();return}const ratings={created_at:new Date().toISOString(),listener_status:'user_entered',scores:{},notes:document.querySelector('#notes').value};document.querySelectorAll('input[data-item]').forEach(i=>{const item=i.dataset.item;ratings.scores[item]??={};ratings.scores[item][i.dataset.dimension]=i.value===''?null:Number(i.value)});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(ratings,null,2)],{type:'application/json'}));a.download='listening_ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};</script></html>'''
    context = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    document = document.replace("const ratings={created_at:", "const ratings={context:" + context + ",created_at:")
    page = destination / "index.html"
    page.write_text(document, encoding="utf-8")
    return page


def score_fields(item: str, dimensions: list[str]) -> str:
    return '<div class="scores">' + ''.join(
        f'<label>{html.escape(d)}（1–10）<input type="number" min="1" max="10" step="1" data-item="{item}" data-dimension="{html.escape(d)}"></label>' for d in dimensions) + '</div>'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--workspace", type=Path, default=ROOT / "data/voice_studio")
    args = parser.parse_args()
    print(build(args.delivery.resolve(), args.workspace.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
