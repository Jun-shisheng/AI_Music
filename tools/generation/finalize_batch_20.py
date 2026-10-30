"""Finalize the bilingual batch with vocal-only cleanup and plain-text QA."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEN_ROOT = PROJECT_ROOT / "tools" / "generation"
sys.path.insert(0, str(GEN_ROOT))

from diagnose_audio import analyze  # noqa: E402

BATCH = PROJECT_ROOT / "outputs" / "batch_20260820"
INPUTS = BATCH / "inputs"
CANDIDATES = BATCH / "candidates"
FINAL = BATCH / "final"
WORK = BATCH / "work"
VOCAL_PYTHON = PROJECT_ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"
SURGICAL = GEN_ROOT / "surgical_vocal_variant.py"


def read_index() -> list[tuple[str, str]]:
    rows = []
    for line in (INPUTS / "INDEX.txt").read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            slug, title, _style = line.split("\t", 2)
            rows.append((slug, title))
    return rows


def basic_pass(report: dict) -> bool:
    return (
        report["duration_s"] >= 29.0
        and report["peak"] >= 0.02
        and report["dropout_total_s"] <= 0.5
    )


def cleanup_is_safe(source: dict, cleaned: dict) -> bool:
    return (
        basic_pass(cleaned)
        and cleaned["hf_energy_ratio"] <= source["hf_energy_ratio"] * 1.05
        and cleaned["dropout_total_s"] <= source["dropout_total_s"] + 0.05
        and cleaned["click_count"] <= source["click_count"] + 8
    )


def make_mp3(wav: Path, mp3: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)],
        check=True,
    )


def main() -> int:
    FINAL.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    lines = ["20-song bilingual batch quality report", ""]
    promoted = 0

    for number, (slug, title) in enumerate(read_index(), start=1):
        source = CANDIDATES / f"{slug}.wav"
        if not source.exists():
            lines.append(f"{number:02d} {title}: MISSING source")
            continue
        source_report = analyze(source)
        cleaned = WORK / f"{slug}_cleaned.wav"
        work_dir = WORK / slug
        selected = source
        mode = "source"
        try:
            subprocess.run(
                [
                    str(VOCAL_PYTHON), str(SURGICAL),
                    "--input", str(source),
                    "--output", str(cleaned),
                    "--work-dir", str(work_dir),
                    "--vocal-noise-profile", "transparent",
                    "--ratio-thresh", "2.5",
                    "--max-cut-db", "-8",
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            cleaned_report = analyze(cleaned)
            if cleanup_is_safe(source_report, cleaned_report):
                selected = cleaned
                mode = "vocal-clean"
                chosen_report = cleaned_report
            else:
                chosen_report = source_report
                mode = "source-fallback"
        except Exception as exc:  # noqa: BLE001
            chosen_report = source_report
            mode = f"source-fallback ({type(exc).__name__})"

        output_stem = f"{number:02d}_{title}"
        final_wav = FINAL / f"{output_stem}.wav"
        shutil.copy2(selected, final_wav)
        make_mp3(final_wav, FINAL / f"{output_stem}.mp3")
        shutil.copy2(INPUTS / f"{slug}_lyrics.txt", FINAL / f"{output_stem}.txt")
        promoted += 1
        lines.append(
            f"{number:02d} {title}: {mode}; duration={chosen_report['duration_s']:.2f}s; "
            f"hf_ratio={chosen_report['hf_energy_ratio']:.4f}; clicks={chosen_report['click_count']}; "
            f"dropout={chosen_report['dropout_total_s']:.2f}s"
        )
        cleaned.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir)

    if WORK.exists() and not any(WORK.iterdir()):
        WORK.rmdir()
    lines.extend(["", f"promoted={promoted}/20"])
    (FINAL / "QUALITY_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"finalized {promoted}/20 into {FINAL}")
    return 0 if promoted == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
