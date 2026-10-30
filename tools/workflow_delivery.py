"""Collect local music outputs into a dated, self-contained listening folder."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
LOCAL = timezone(timedelta(hours=8))
LEGACY = ROOT / "_archive/outputs_history/2026-10-19_quality_upgrade"
FINISHED = {"awaiting_listening", "technical_review_required", "failed"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def lyric_lines(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip() and not re.fullmatch(r"\[.*\]", line.strip()))


class Collector:
    def __init__(self, destination: Path):
        self.destination = destination.resolve()
        self.destination.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, dict] = load(self.destination / "package_manifest.json").get("archive_files", {})
        for relative, record in self.files.items():
            target = (self.destination / relative).resolve()
            if (not target.is_relative_to(self.destination) or not target.is_file()
                    or sha(target) != record.get("sha256")):
                raise ValueError(f"Existing delivery archive is missing or changed: {relative}")

    def copy(self, source: Path, relative: Path, expected: str | None = None) -> Path:
        source = source.resolve()
        target = (self.destination / relative).resolve()
        if not target.is_relative_to(self.destination):
            raise ValueError("Delivery target escapes its date folder")
        digest = sha(source)
        if expected and digest != expected:
            raise ValueError(f"Source SHA mismatch: {source}")
        if target == source:
            pass
        elif target.is_file():
            if sha(target) != digest:
                raise ValueError(f"Archived file changed; preserve it and use a new version: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha(target) != digest:
                raise OSError(f"Copy verification failed: {target}")
        self.files[target.relative_to(self.destination).as_posix()] = {
            "source": str(source), "sha256": digest, "bytes": target.stat().st_size}
        return target

    def tree(self, source: Path, relative: Path) -> Path:
        source = source.resolve()
        target = (self.destination / relative).resolve()
        if source == target or target.is_relative_to(source):
            raise ValueError("Cannot snapshot a directory inside itself")
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Linked archive input requires an explicit source: {path}")
            if path.is_file() and not path.name.endswith(".tmp"):
                self.copy(path, relative / path.relative_to(source))
        return target


def verified_outputs(manifest: dict) -> bool:
    outputs = manifest.get("outputs", {})
    if not outputs:
        return manifest.get("status") == "failed"
    return all(Path(value.get("path", "__missing__")).is_file()
               and sha(Path(value["path"])) == value.get("sha256") for value in outputs.values())


def legacy_snapshot(collector: Collector, source: Path) -> Path:
    base = Path("archive/initial_quality_upgrade")
    prefix = base.as_posix() + "/"
    existing = sorted((name[len(prefix):], item["sha256"]) for name, item in collector.files.items() if name.startswith(prefix))
    current = sorted((path.relative_to(source).as_posix(), sha(path)) for path in source.rglob("*")
                     if path.is_file() and not path.name.endswith(".tmp"))
    if existing and existing != current:
        fingerprint = hashlib.sha256(json.dumps(current, ensure_ascii=False).encode()).hexdigest()[:12]
        base = Path("archive") / f"initial_quality_upgrade_{fingerprint}"
    collector.tree(source, base)
    return base


def snapshot_code(collector: Collector, repo: Path) -> str | None:
    files = [repo / path for path in ["run_workflows.bat", "tools/music_workflows.py", "tools/music_qa.py",
             "tools/workflow_delivery.py", "tools/workflow_review.py", "docs/quality_upgrade_workflows.md",
             "docs/arrangement_spec.md", "progress.md"]]
    for name in ["voice_studio", "reference_studio", "tests"]:
        files.extend(path for path in (repo / "tools" / name).rglob("*") if path.is_file()
                     and path.suffix in {".py", ".md", ".txt", ".json", ".abc"} and "__pycache__" not in path.parts)
    files = sorted(set(path for path in files if path.is_file()))
    if not files:
        return None
    entries = [(path.relative_to(repo).as_posix(), sha(path)) for path in files]
    identity = hashlib.sha256(json.dumps(entries, ensure_ascii=False).encode()).hexdigest()[:16]
    relative = Path("archive/code") / identity
    for path, (_, digest) in zip(files, entries):
        collector.copy(path, relative / path.relative_to(repo), digest)
    return relative.as_posix()


def collect(destination: Path, workspace: Path, legacy: Path = LEGACY, repo: Path = ROOT) -> Path:
    collector = Collector(destination)
    warnings, cards, jobs, diagnostics = [], [], {}, []
    legacy_relative = Path("archive/initial_quality_upgrade")
    if legacy.is_dir():
        legacy_relative = legacy_snapshot(collector, legacy)
    for path in sorted((workspace / "profiles").glob("*/profile.json")):
        collector.tree(path.parent, Path("archive/profiles") / f"{path.parent.name}_{sha(path)[:12]}")
    for path in sorted((workspace / "jobs").glob("*/manifest.json")):
        data = load(path)
        if data.get("status") not in FINISHED:
            continue
        if data.get("status") == "failed":
            snapshot = collector.tree(path.parent, Path("archive/diagnostics") / f"{path.parent.name}_{sha(path)[:12]}")
            diagnostics.append(snapshot.relative_to(collector.destination).as_posix())
            continue
        if not verified_outputs(data):
            warnings.append(f"音频摘要失配，未收录：{path.parent.name}")
            continue
        snapshot = collector.tree(path.parent, Path("archive/voice_jobs") / f"{path.parent.name}_{sha(path)[:12]}")
        jobs[path.parent.name] = (data, snapshot)
    for path in sorted((repo / "logs/music_workflows").glob("*/run.json")):
        if path.parent.name.startswith(destination.name.replace("-", "")) and load(path).get("status") != "running":
            collector.tree(path.parent, Path("archive/run_logs") / path.parent.name)
    for relative in ["docs/quality_upgrade_workflows.md", "tools/voice_studio/README.md", "tools/reference_studio/README.md"]:
        source = repo / relative
        if source.is_file():
            collector.copy(source, Path("archive/instructions") / f"{source.parent.name}_{source.stem}_{sha(source)[:12]}.md")
    code_snapshot = snapshot_code(collector, repo)
    for source in sorted((repo / "inputs/workflows").glob("*")):
        if source.is_file() and source.suffix.lower() in {".txt", ".json", ".abc"}:
            collector.copy(source, Path("archive/workflow_inputs") / f"{source.stem}_{sha(source)[:12]}{source.suffix}")

    audio_records = []
    quality = {}
    for path in sorted((collector.destination / "qa").rglob("qa.json")):
        result, request = load(path), load(path.parent / "request.json")
        record = load(path.parent / "qa_manifest.json")
        expected = record.get("outputs", {})
        if (result.get("status") == record.get("status") == "complete"
                and result.get("request_identity") == record.get("request_identity")
                and sha(path) == expected.get("qa.json")
                and (path.parent / "request.json").is_file()
                and sha(path.parent / "request.json") == expected.get("request.json")):
            quality[(request.get("audio", {}).get("sha256"), tuple(result.get("target_lines", [])))] = (result, path.parent)

    def audio(label: str, source: Path, key: str, expected: str | None = None, qa_sha: str | None = None,
              target_lines: tuple[str, ...] | None = None) -> str:
        if not source.is_file():
            return '<p class="pending">等待生成</p>'
        # Versioned names keep previously exported ratings attributable after another run.
        digest = sha(source)
        copied = collector.copy(source, Path("audio") / f"{key}_{digest[:12]}{source.suffix.lower()}", expected)
        relative = copied.relative_to(destination.resolve()).as_posix()
        item = {"label": label, "path": relative, "sha256": digest}
        details = ''
        qa_key = (qa_sha or digest, target_lines)
        if qa_key in quality:
            result, directory = quality[qa_key]
            item["qa_request_identity"] = result["request_identity"]
            comparison = result.get("comparison", {})
            edits = comparison.get("word_edits", {})
            summary = f'替换{edits.get("substitution", 0)}，漏词{edits.get("deletion", 0)}，插词{edits.get("insertion", 0)}'
            if comparison.get("lines_total"):
                summary = f'近似恢复{comparison.get("approximate_lines_recovered", 0)}/{comparison["lines_total"]}句；' + summary
            details = '<details><summary>自动歌词检查：' + html.escape(summary) + '</summary><p>ASR用于定位问题，不能替代听审。</p><pre>' + html.escape(result.get("transcript", {}).get("text", '')) + '</pre></details>'
        audio_records.append(item)
        return f'<label>{html.escape(label)}</label><audio controls preload="none" src="{quote(relative)}"></audio>' + details

    profile = load(workspace / "profiles/iron_rain_female_v1/profile.json")
    anchor = profile.get("artifacts", {}).get("reference", {})
    if anchor:
        cards.append('<section><h2>声音母本 · Iron Rain</h2><p>固定参考女声，7秒干声。</p>' + audio("女声参考", Path(anchor["path"]), "00_voice_anchor", anchor["sha256"]) + '</section>')
    titles = {"cover_no_white_flag_v1": "No White Flag", "cover_hammer_v1": "Hammer and Hymn",
              "cover_no_white_flag_full_v1": "No White Flag · 全曲"}
    priority = {"cover_no_white_flag_v1": 0, "cover_hammer_v1": 1, "cover_no_white_flag_full_v1": 2}
    batch_titles = load(repo / "inputs/2026-10-15_epic_en/_titles.json") if (repo / "inputs/2026-10-15_epic_en/_titles.json").is_file() else {}
    for name in sorted(jobs, key=lambda value: (priority.get(value, 2), value)):
        data, snapshot = jobs[name]
        if data.get("inputs", {}).get("command") != "cover":
            continue
        title = titles.get(name, name)
        match = re.fullmatch(r"cover_(.+)_full_v(\d+)", name)
        slug = match.group(1) if match else None
        if slug == "no_white_flag":
            slug = "02_no_white_flag"
        if match:
            title = f"{batch_titles.get(slug, slug)} · 全曲" + ("" if match.group(2) == "1" else f" v{match.group(2)}")
        outputs = data.get("outputs", {})
        content = audio("原演唱", snapshot / "original.wav", name + "_original")
        lyrics_file = repo / "inputs/2026-10-15_epic_en" / f"{slug}_lyrics.txt" if slug else None
        target_lines = lyric_lines(lyrics_file.read_text(encoding="utf-8")) if lyrics_file and lyrics_file.is_file() else None
        for key, label in [("mix", "固定女声与伴奏"), ("vocals", "固定女声干声")]:
            if key in outputs:
                content += audio(label, snapshot / Path(outputs[key]["path"]).name, name + "_" + key,
                                 outputs[key]["sha256"], target_lines=target_lines if key == "vocals" else None)
        if match:
            note = ("全曲固定女声版；内容代理与原始演唱的对照见 finished/README.txt。"
                    + (" 替代 take，与 full_v1 一起听。" if match.group(2) != "1" else ""))
        else:
            note = ("词句保留较稳定，可优先试听。" if name == "cover_no_white_flag_v1" else
                    ("词句保留需重点复核。" if name == "cover_hammer_v1" else ''))
        cards.append(f'<section><h2>固定女声翻唱 · {html.escape(title)}</h2><p>{note}音色身份和自然度等待试听。</p>{content}</section>')

    content = ''
    for path in sorted((collector.destination / "experiments/voice").glob("*/manifest.json")):
        data = load(path)
        if data.get("kind") != "phrase_comp" or data.get("status") not in {"awaiting_listening", "technical_review_required"} or not verified_outputs(data):
            continue
        item = data["outputs"]["vocals"]
        lyric_file = path.parent / "intended_lyrics.txt"
        lyrics = lyric_file.read_text(encoding="utf-8") if lyric_file.is_file() else ''
        content += '<article><h3>新副歌 · 逐句组合候选</h3><p>组合不同版本的句子，请重点检查接缝、咬字和音色。</p><pre>' + html.escape(lyrics) + '</pre>' + audio("组合后的固定女声干声", Path(item["path"]), path.parent.name, item["sha256"], target_lines=lyric_lines(lyrics)) + '</article>'
    for name in sorted(jobs, key=lambda value: -int(re.search(r"_v(\d+)$", value).group(1)) if re.search(r"_v(\d+)$", value) else 0):
        data, snapshot = jobs[name]
        if data.get("inputs", {}).get("command") == "relyric" and "vocals" in data.get("outputs", {}):
            item = data["outputs"]["vocals"]
            lyric_file = snapshot / "intended_lyrics.txt"
            lyrics = lyric_file.read_text(encoding="utf-8") if lyric_file.is_file() else ''
            version = re.search(r"_v(\d+)$", name)
            title = "新副歌 · 第" + version.group(1) + "版" if version else name
            content += '<article><h3>' + html.escape(title) + '</h3><pre>' + html.escape(lyrics) + '</pre>' + audio("固定女声干声", snapshot / Path(item["path"]).name, name, item["sha256"], target_lines=lyric_lines(lyrics)) + '</article>'
    if content:
        cards.append('<section><h2>改词重唱</h2><p>版本号表示修订次序。咬字仍在精修，不能只按版本号判断听感好坏。</p>' + content + '</section>')

    experiments = []
    initial = collector.destination / legacy_relative / "reference/iron_rain_support"
    if initial.is_dir():
        experiments.append(initial)
    experiments.extend(sorted((collector.destination / "experiments/reference").glob("*/manifest.json")))
    for item in experiments:
        folder = item.parent if item.is_file() else item
        report = load(folder / "report.json")
        manifest = load(folder / "manifest.json")
        if manifest.get("kind") == "controlled_arrangement_prepare":
            links, assets, voices = [], [], []
            variants = manifest.get("variants", {})
            if manifest.get("status") != "prepared_not_rendered" or set(variants) != {"sparse", "expanded"}:
                continue
            valid = True
            for variant, label in [("sparse", "稀疏版"), ("expanded", "展开版")]:
                for key, title in [("midi", "MIDI"), ("events", "四轨音符事件")]:
                    item = variants[variant].get(key, {})
                    source = (folder / item.get("path", "__missing__")).resolve()
                    if not source.is_relative_to(folder.resolve()) or not source.is_file() or sha(source) != item.get("sha256"):
                        valid = False
                        break
                    assets.append((source, item["sha256"]))
                    links.append('<a href="' + quote(source.relative_to(collector.destination).as_posix()) + '">' + label + ' ' + title + '</a>')
                voice = variants[variant].get("vocal_reference", {})
                source = (folder / voice.get("path", "__missing__")).resolve()
                if not source.is_relative_to(folder.resolve()) or not source.is_file() or sha(source) != voice.get("sha256"):
                    valid = False
                voices.append(voice.get("sha256"))
            if not valid or len(set(voices)) != 1:
                warnings.append("可控编曲素材摘要失配，未展示：" + folder.name)
                continue
            for source, digest in assets:
                collector.copy(source, source.relative_to(collector.destination), digest)
            cards.append('<section><h2>可控编曲 · MIDI分轨素材</h2><p>稀疏、展开两版共用同一份固定女声，鼓、贝斯、和弦与弦乐分别编排。此处是工程素材，尚未渲染伴奏或混音。</p><p>' + ' · '.join(links) + '</p></section>')
            continue
        content, shown, failed, timing_warnings = '', 0, [], []
        for variant, label in [("baseline", "基准版"), ("arranged", "参考编曲版")]:
            state = load(folder / f"renders/{variant}/state.json")
            if state.get("status") == "failed":
                failed.append(label)
            native = folder / f"renders/{variant}/audio.wav"
            if state.get("status") != "complete" or not native.is_file() or sha(native) != state.get("audio_sha256"):
                continue
            timing = report.get("metrics", {}).get(variant, {}).get("duration_validation", {})
            if (report.get("experiment_identity") == manifest.get("experiment_identity")
                    and timing.get("technical_review_required")):
                timing_warnings.append(
                    f'{label}实际{timing["actual_duration_s"]:.3f}秒，'
                    f'相对计划偏差{timing["relative_error_percent"]:+.1f}%；未通过时长检查，保留试听诊断。')
            selected, expected = native, state["audio_sha256"]
            matched = (report.get("loudness_match") or {}).get("files", {}).get(variant, {})
            normalized = folder / f"listening/{variant}.wav"
            if (report.get("experiment_identity") == manifest.get("experiment_identity") and
                    matched.get("source_sha256") == expected and normalized.is_file() and
                    sha(normalized) == matched.get("listening_sha256")):
                selected, expected = normalized, matched["listening_sha256"]
            else:
                label += "（原始响度）"
            target = folder / "lyrics.txt"
            lines = lyric_lines(target.read_text(encoding="utf-8")) if target.is_file() else None
            content += audio(label, selected, folder.name + "_" + variant, expected, state["audio_sha256"], lines)
            shown += 1
        if content:
            title = "完整主歌进入副歌" if folder == initial else ("副歌短段对照" if shown == 2 else "副歌单版实验")
            flag = '<p class="pending">' + '、'.join(failed) + '生成未通过检查，诊断记录已保留。</p>' if failed else ''
            flag += ''.join('<p class="pending">' + html.escape(warning) + '</p>' for warning in timing_warnings)
            cards.append('<section><h2>参考编曲 · ' + title + '</h2><p>比较段落对比、律动与人声；未进行主观听审。</p>' + flag + content + '</section>')

    stamp = datetime.now(LOCAL).isoformat(timespec="seconds")
    catalog = {"created_at": stamp, "status": "pending_user_listening", "automatic_music_score": None,
               "date_folder": str(collector.destination), "audio": audio_records, "warnings": warnings,
               "latest_legacy_snapshot": legacy_relative.as_posix() if legacy.is_dir() else None,
               "failed_job_diagnostics": diagnostics,
               "code_snapshot": code_snapshot,
               "archive_files": collector.files,
               "note": "Daily copies are self-contained. Persistent singer profiles and job originals remain in data/voice_studio."}
    save(collector.destination / "package_manifest.json", catalog)
    context = json.dumps({key: catalog[key] for key in ["created_at", "status", "audio"]}, ensure_ascii=False).replace("<", "\\u003c")
    legacy_link = (f' · <a href="{quote((legacy_relative / "listening/index.html").as_posix())}">首轮完整对照页</a>'
                   if (collector.destination / legacy_relative / "listening/index.html").is_file() else '')
    finished_link = (' · <a href="finished/">整曲成品文件夹</a>'
                     if (collector.destination / "finished").is_dir() else '')
    notes_link = ' · <a href="ROUND_NOTES.txt">本轮进展说明</a>' if (collector.destination / "ROUND_NOTES.txt").is_file() else ''
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Music · 日期试听集</title>
<style>body{margin:0;background:#101622;color:#e6edf6;font:16px/1.65 system-ui,sans-serif}main{max-width:1020px;margin:auto;padding:30px 22px}h1{font-size:30px}h2{font-size:21px;margin-top:0}p{color:#bac5d6}section{margin:20px 0;padding:24px;border:1px solid #35445b;border-radius:15px;background:#192334}audio{display:block;width:100%;margin:8px 0 20px}label{color:#d9e7fb}pre{white-space:pre-wrap;color:#bad8d0}textarea{box-sizing:border-box;width:100%;padding:12px;border:1px solid #526884;background:#111a29;color:#fff;border-radius:8px}button{padding:12px 20px;background:#96d2be;color:#102b22;border:0;border-radius:8px;cursor:pointer}a{color:#a6c9ff}.pending{color:#e9c28a}</style>
<main><h1>AI Music · ''' + html.escape(collector.destination.name) + '''</h1><p>先听女声母本，再比较翻唱、新词和参考编曲。版本号用于追踪修改，听感尚待你确认。</p><p>所有可播放音频均在本日期文件夹内。生成：''' + stamp + '</p>' + ''.join('<p class="pending">' + html.escape(item) + '</p>' for item in warnings) + ''.join(cards) + '''<section><h2>试听记录</h2><textarea id="notes" rows="5" placeholder="写歌名或版本、时间点，以及你听到的问题或偏好"></textarea><p><button id="save">导出试听记录</button></p><p><a href="package_manifest.json">文件与音频清单</a>''' + legacy_link + finished_link + notes_link + '''</p></section></main>
<script>const context=''' + context + ''';document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));document.querySelector('#save').onclick=()=>{const data={context,recorded_at:new Date().toISOString(),notes:document.querySelector('#notes').value};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='music_listening_notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};</script></html>'''
    page = collector.destination / "index.html"
    page.write_text(document, encoding="utf-8")
    (collector.destination / "README.txt").write_text(
        "双击 index.html 开始试听。\n\n音频：audio\n历史完整产出、歌手档案、推理任务、源码和说明副本：archive\n新实验：experiments\n自动检查：qa\n\n自动检查不替代音乐听感与音色验收。试听记录可在页面导出。\n", encoding="utf-8-sig")
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(LOCAL).date().isoformat())
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--workspace", type=Path, default=ROOT / "data/voice_studio")
    args = parser.parse_args()
    try:
        day = date.fromisoformat(args.date)
        if day.isoformat() != args.date:
            raise ValueError("Date must use YYYY-MM-DD")
        page = collect((args.outputs / day.isoformat()).resolve(), args.workspace.resolve())
        print(page)
        return 0
    except (OSError, ValueError) as error:
        print(f"Delivery failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
