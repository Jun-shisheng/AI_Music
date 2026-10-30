"""Build the high-note A/B listening set and its evidence table.

The experiment is deliberately small: two archived YuE2 takes, a window that
contains a measured high note, and the same window through the legacy 24 kHz
SVC path and the new hybrid path. All listening files get one constant gain to
-20 dBFS so loudness cannot masquerade as quality. Metrics are descriptive.
"""
from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.voice_studio.highnote_qa import analyze  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float64", always_2d=False)
    return data, int(rate)


def _resample(data: np.ndarray, source: int, target: int = 44100) -> np.ndarray:
    if source == target:
        return data
    common = gcd(source, target)
    return resample_poly(data, target // common, source // common, axis=0)


def _level_match(data: np.ndarray, rate: int, target_dbfs: float = -20.0) -> tuple[np.ndarray, dict]:
    mono = data.mean(axis=1) if data.ndim > 1 else data
    rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
    if rms <= 0:
        return data, {"input_dbfs": None, "gain_db": 0.0}
    input_dbfs = 20.0 * np.log10(rms)
    gain_db = target_dbfs - input_dbfs
    output = data * (10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(output))) if len(output) else 0.0
    if peak > 0.95:
        headroom = 20.0 * np.log10(0.95 / peak)
        output = output * (10.0 ** (headroom / 20.0))
        gain_db += headroom
    return output, {"input_dbfs": round(input_dbfs, 2), "gain_db": round(gain_db, 2)}


def _write_listening(source: Path, target: Path, target_dbfs: float) -> dict:
    data, rate = _read(source)
    data = _resample(data, rate, 44100)
    data, report = _level_match(data, 44100, target_dbfs)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), data, 44100, subtype="PCM_16")
    return {"source": str(source), "output": str(target), "rate": 44100, **report}


def _job_output(job: Path, key: str) -> tuple[Path, int]:
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    path = Path(manifest["outputs"][key]["path"])
    info = sf.info(str(path))
    return path, int(info.samplerate)


def build(config: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "listening").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    target_dbfs = float(config.get("target_dbfs", -20.0))
    rows = []
    for song in config["songs"]:
        name = song["name"]
        native_mix = _resolve(song["native_mix"])
        native_vocals = _resolve(song["native_vocals"])
        native_data, native_rate = _read(native_vocals)
        arms = []
        arms.append(("native", native_mix, native_vocals))
        legacy = _resolve(song["legacy_job"])
        hybrid = _resolve(song["hybrid_job"])
        arms.append(("legacy24", *_job_output(legacy, "mix")))
        arms.append(("hybrid", *_job_output(hybrid, "mix")))
        arms[1] = ("legacy24", arms[1][1], _job_output(legacy, "vocals")[0])
        arms[2] = ("hybrid", arms[2][1], _job_output(hybrid, "vocals")[0])

        for arm, mix, vocal in arms:
            mix_out = _write_listening(mix, out_dir / "listening" / f"{name}_{arm}_mix.wav", target_dbfs)
            vocal_out = _write_listening(vocal, out_dir / "listening" / f"{name}_{arm}_vocal.wav", target_dbfs)
            vocal_data, vocal_rate = _read(vocal)
            if vocal_data.ndim > 1:
                vocal_data = vocal_data.mean(axis=1)
            qa = analyze(vocal_data, vocal_rate, source=native_data, source_rate=native_rate)
            (out_dir / "metrics" / f"{name}_{arm}_highnote.json").write_text(
                json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            rows.append({"song": name, "arm": arm, "mix": mix_out, "vocal": vocal_out, "qa": qa})
    manifest = {"schema": 1, "target_dbfs": target_dbfs,
                "status": "awaiting_listening", "promotion_blocked": True,
                "songs": config["songs"], "rows": rows}
    (out_dir / "ab_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(out_dir, config, rows)
    return manifest


def _write_report(out_dir: Path, config: dict, rows: list[dict]) -> None:
    lines = ["# 高音/转音 A/B 试听报告", "",
             "所有文件统一常数增益到 -20 dBFS；没有压缩、限幅或动态处理。"
             "先听 `_vocal.wav`（只看人声），再听 `_mix.wav`。", "",
             "自动指标只描述，不证明好听；最终以听感为准。", "",
             "| 歌曲 | 版本 | F0 相关 | 八度错误 | HF 比 | 高音段 | 标记 |",
             "|---|---|---:|---:|---:|---|---|"]
    for row in rows:
        qa = row["qa"]
        comparison = qa.get("comparison") or {}
        flags = ", ".join(qa.get("flags", [])) or "-"
        segments = qa.get("high_segments", [])
        segment_text = "<br>".join(
            f"{s['start_s']:.2f}-{s['end_s']:.2f}s {s['median_f0_hz']:.0f}Hz "
            f"vib={s['vibrato_depth_cents']}c HNR={s['hnr_db']}dB"
            for s in segments[:3]) or "-"
        lines.append(
            f"| {row['song']} | {row['arm']} | "
            f"{comparison.get('f0_contour_correlation', '-')} | "
            f"{comparison.get('octave_error_ratio', '-')} | "
            f"{comparison.get('hf_ratio_11k', '-')} | {segment_text} | {flags} |")
    lines += ["", "## 试听顺序", ""]
    for song in config["songs"]:
        lines.append(f"### {song['name']}")
        if song.get("high_notes"):
            lines.append(f"高音位置：{song['high_notes']}")
        lines.append("1. native 人声 → 2. legacy24 人声 → 3. hybrid 人声")
        lines.append("4. native 混音 → 5. legacy24 混音 → 6. hybrid 混音")
        lines.append("")
    lines += ["## 判定", "",
              "- hybrid 只有在高音/转音上明显比 legacy24 自然、且不输给 native 时，才允许升格为交付路径。",
              "- 若 hybrid 仍机械，则固定女声线默认交付 native（YuE2 原生 48 kHz），SVC 只保留为实验。",
              "- 参考编曲线的计划整形与 MIDI 段落另开一轮，不和本轮的换声结论混在一起。", ""]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "试听顺序.txt").write_text(
        "\n".join(lines[lines.index("## 试听顺序") + 1:lines.index("## 判定")]).strip() + "\n",
        encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = (args.out or args.config.parent).resolve()
    manifest = build(config, out_dir)
    print(json.dumps({"status": manifest["status"], "out": str(out_dir),
                      "rows": len(manifest["rows"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
