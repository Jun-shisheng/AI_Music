"""Package an epic/ethereal YuE2 batch for the private Music_date repo.

Layout follows the 2026-10-26 delivery (pure lyrics, no header; manifest
schema 1; README in Chinese):

    <repo>/2026-10-28_zh_epic/
      中文女声/<slug>_YuE2.wav + <slug>_YuE2.txt
      器乐/<slug>_YuE2.wav + <slug>_YuE2.txt
      manifest.json
      README.md
      review.md

Usage:
  python -I tools/generation/package_epic_delivery.py \
      --inputs inputs/2026-10-28_epic_batch \
      --batch-dir outputs/2026-10-28_epic_batch \
      --name 2026-10-28_zh_epic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]

STYLE_LABEL = {
    "epic": "哲思史诗（御姐）",
    "epic_ethereal": "空灵史诗",
    "battle_anthem": "战斗史诗",
    "instrumental": "器乐 BGM",
}


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_stats(path: Path) -> dict:
    info = sf.info(str(path))
    data, _ = sf.read(str(path), always_2d=True, dtype="float32")
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(data ** 2))) if data.size else 0.0
    return {"duration_seconds": round(info.frames / info.samplerate, 2),
            "sample_rate": info.samplerate, "channels": info.channels,
            "peak": round(peak, 6), "rms": round(rms, 6)}


def collect(inputs: Path, batch: Path) -> list[dict]:
    modes = load(inputs / "_modes.json", {})
    briefs = load(inputs / "briefs.json", {})
    rows: list[dict] = []
    groups = [(batch, "中文女声")]
    if (batch / "bgm").exists():
        groups.append((batch / "bgm", "器乐"))
    for root, voice in groups:
        for result in sorted(root.glob("candidates/*/result.json")):
            if "_artifacts" in str(result):
                continue
            data = load(result, {})
            if "arc" not in data:
                continue
            slug = data["title"]
            wav = root / "selected" / f"{slug}.wav"
            if not wav.exists():
                print(f"WARN missing {wav}")
                continue
            brief = briefs.get(slug, {}).get("brief", {})
            arc = data["arc"]
            rows.append({
                "slug": slug,
                "title": brief.get("title", slug),
                "hook": brief.get("hook", ""),
                "lyrics": brief.get("lyrics", ""),
                "mode": modes.get(slug, "instrumental" if root is not batch else ""),
                "voice_dir": voice,
                "wav": wav,
                "arc": arc,
                "coverage": data.get("diction"),
                "melody": data.get("melody") or {},
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=ROOT / ".cache" / "Music_date_repo")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    if not (args.repo / ".git").exists():
        print(f"ERROR: no git clone at {args.repo}")
        return 1
    name = args.name or f"{date.today().isoformat()}_zh_epic"
    out = args.repo / name
    if out.exists():
        shutil.rmtree(out)
    rows = collect(args.inputs, args.batch_dir)
    if not rows:
        print("nothing to package")
        return 1

    songs = []
    for row in rows:
        folder = out / row["voice_dir"]
        folder.mkdir(parents=True, exist_ok=True)
        wav_dst = folder / f"{row['slug']}_YuE2.wav"
        txt_dst = folder / f"{row['slug']}_YuE2.txt"
        shutil.copy2(row["wav"], wav_dst)
        lyrics = (row["lyrics"] or "").strip()
        if row["mode"] == "instrumental" or not lyrics:
            lyrics = "（器乐，无人声。）"
        txt_dst.write_text(lyrics + "\n", encoding="utf-8")
        stats = audio_stats(wav_dst)
        arc = row["arc"]
        songs.append({
            "slug": row["slug"],
            "title": row["title"],
            "hook": row["hook"],
            "language": "zh" if row["mode"] != "instrumental" else "instrumental",
            "voice": row["voice_dir"],
            "style": STYLE_LABEL.get(row["mode"], row["mode"]),
            "engine": "yue2-3b",
            "file": f"{row['voice_dir']}/{wav_dst.name}",
            "lyrics": txt_dst.name,
            **stats,
            "arc_verdict": arc.get("verdict"),
            "lift_db": arc.get("lift_db"),
            "peak_at": arc.get("peak_at"),
            "dip_before_peak_db": arc.get("dip_before_peak_db"),
            "lyric_coverage": row["coverage"],
            "weak_lyrics": bool(row["coverage"] is not None and row["coverage"] < 0.75),
            "melody_score": row["melody"].get("score"),
            "density_ratio": row["melody"].get("density_ratio"),
            "register_lift": row["melody"].get("register_lift"),
            "sha256": sha256(wav_dst),
            "lyrics_sha256": sha256(txt_dst),
        })
    songs.sort(key=lambda s: s["slug"])

    manifest = {
        "schema": 1,
        "date": date.today().isoformat(),
        "engine": "yue2-3b",
        "line": "native_yue2_48k",
        "delivery_status": "awaiting_listening",
        "note": ("2026-10-28 epic/ethereal calibration + bulk batch. Native YuE2 "
                 "48 kHz, no SVC/EQ/compress/limit. One instrumental cue uses "
                 "the TSFH LoRA. All takes are unvetted candidates: the user "
                 "listens and picks."),
        "counts": {voice: sum(1 for s in songs if s["voice"] == voice)
                   for voice in sorted({s["voice"] for s in songs})},
        "songs": songs,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    review_src = args.batch_dir / "review.md"
    if review_src.exists():
        shutil.copy2(review_src, out / "review.md")
    feedback_src = args.batch_dir / "FEEDBACK_MAP.md"
    if feedback_src.exists():
        shutil.copy2(feedback_src, out / "FEEDBACK_MAP.md")

    lines = [
        f"# {name}",
        "",
        "2026-10-28 中文史诗/空灵批。全部 YuE2 原生 48 kHz；无 SVC / EQ / 压缩 / 限幅。",
        "器乐 BGM 使用 TSFH LoRA（AR 0.5 / NAR 0.375）。",
        "",
        "## 目录",
        "",
    ]
    for voice, count in manifest["counts"].items():
        lines.append(f"- `{voice}/`：{count} 首，WAV + 同名纯歌词 TXT")
    lines += [
        "",
        "## 听审",
        "",
        "先看 `review.md` 的表格；过线的先听，待修的看原因。",
        "`钉子`、`举火的人`、`灯亮着` 是自动指标最稳的三首。",
        "",
        "## 自动指标说明",
        "",
        "- 结构：build-up = 后段高潮（欲扬先抑）；early peak = 高潮过早。",
        "- 咬字：whisper 对参考歌词的覆盖，<0.75 标 weak_lyrics。",
        "- 旋律：YuE2 ABC 计划的副歌/主歌密度比、音区抬升、钩子相似度。",
        "- 指标只负责拦坏文件，不代替听审。",
        "",
        "## 状态",
        "",
        "待听审。未筛选，全部候选都在。",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")

    subprocess.run(["git", "add", "-A"], cwd=args.repo, check=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=args.repo,
                            capture_output=True, text=True, encoding="utf-8")
    if not status.stdout.strip():
        print("nothing new to commit")
        return 0
    message = (f"{name}: {len(songs)} tracks (YuE2, epic/ethereal)\n\n"
               + "\n".join(f"- {s['slug']} {s['title']}" for s in songs))
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=args.repo, check=True)
    if args.no_push:
        print(f"committed locally ({len(songs)} tracks); --no-push")
        return 0
    push = subprocess.run(["git", "push", "origin", "main"], cwd=args.repo,
                          capture_output=True, text=True, encoding="utf-8")
    if push.returncode != 0:
        print(f"PUSH FAILED rc={push.returncode}\n{push.stderr[-800:]}")
        return 1
    print(f"PUSHED {name}: {len(songs)} tracks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
