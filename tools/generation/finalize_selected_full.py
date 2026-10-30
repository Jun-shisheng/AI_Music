"""Vocal-only cleanup and packaging for the selected full-length songs."""
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
CANDIDATES = BATCH / "full_candidates"
FINAL = BATCH / "full_final"
WORK = BATCH / "full_work"
VOCAL_PYTHON = PROJECT_ROOT / "tools" / "vocal_repair" / ".venv" / "Scripts" / "python.exe"
SURGICAL = GEN_ROOT / "surgical_vocal_variant.py"


def read_index() -> dict[str, tuple[int, str]]:
    rows = {}
    for line in (INPUTS / "INDEX.txt").read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            slug, title, _style = line.split("\t", 2)
            rows[slug] = (int(slug[:2]), title)
    return rows


def make_mp3(wav: Path, mp3: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)], check=True)


def safe(source: dict, cleaned: dict) -> bool:
    return (cleaned["duration_s"] >= 150 and cleaned["peak"] >= 0.02
            and cleaned["dropout_total_s"] <= source["dropout_total_s"] + 0.05
            and cleaned["hf_energy_ratio"] <= source["hf_energy_ratio"] * 1.05
            and cleaned["click_count"] <= source["click_count"] + 8)


def main() -> int:
    FINAL.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rows = read_index()
    report = ["Selected full-song quality report", ""]
    count = 0
    for source in sorted(CANDIDATES.glob("*.wav")):
        number, title = rows[source.stem]
        source_report = analyze(source)
        cleaned = WORK / f"{source.stem}_cleaned.wav"
        work_dir = WORK / source.stem
        selected, mode, chosen = source, "source", source_report
        try:
            subprocess.run([str(VOCAL_PYTHON), str(SURGICAL), "--input", str(source), "--output", str(cleaned), "--work-dir", str(work_dir), "--vocal-noise-profile", "transparent", "--ratio-thresh", "2.5", "--max-cut-db", "-8"], cwd=PROJECT_ROOT, check=True)
            cleaned_report = analyze(cleaned)
            if safe(source_report, cleaned_report):
                selected, mode, chosen = cleaned, "vocal-clean", cleaned_report
            else:
                mode = "source-fallback"
        except Exception as exc:  # noqa: BLE001
            mode = f"source-fallback ({type(exc).__name__})"
        stem = f"{number:02d}_{title}"
        final_wav = FINAL / f"{stem}.wav"
        shutil.copy2(selected, final_wav)
        make_mp3(final_wav, FINAL / f"{stem}.mp3")
        shutil.copy2(INPUTS / f"{source.stem}_lyrics.txt", FINAL / f"{stem}.txt")
        report.append(f"{number:02d} {title}: {mode}; duration={chosen['duration_s']:.2f}s; hf_ratio={chosen['hf_energy_ratio']:.4f}; clicks={chosen['click_count']}; dropout={chosen['dropout_total_s']:.2f}s")
        count += 1
        cleaned.unlink(missing_ok=True)
        if work_dir.exists():
            shutil.rmtree(work_dir)
    if WORK.exists() and not any(WORK.iterdir()):
        WORK.rmdir()
    report.extend(["", f"packaged={count}/{len(list(CANDIDATES.glob('*.wav')))}"])
    (FINAL / "QUALITY_REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0 if count == 13 else 1


if __name__ == "__main__":
    raise SystemExit(main())
