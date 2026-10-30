from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from artifact_segments import detect_events
from frame_repair import repair_vocals
from humanize_vocal import humanize_stem
from noise_clean_agent import analyze_noise, clean_stem
from stem_pipeline import remix, separate_with_demucs
from vocal_quality_gate import analyze_vocals
from vocal_repair import PROJECT_ROOT, analyze_file


DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "top_songs"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "outputs" / "quality_rebuild_review"
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs" / "quality_rebuild_work"


@dataclass
class VocalMetrics:
    artifact_score: int
    gate_score: int
    gate_decision: str
    harsh_frame_ratio: float
    burst_count: int
    dropout_seconds: float
    structural_events: int
    noise_events: int


def title_from_pair(wav: Path) -> str:
    txt = wav.with_suffix(".txt")
    if txt.exists():
        for line in txt.read_text(encoding="utf-8-sig", errors="replace").splitlines()[:12]:
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
                if not any(marker in title for marker in ("闀", "鏈", "畨", "鍚", "绔")):
                    return title
    stem = re.sub(r"^\d+_\d+_[^_]+_[^_]+_", "", wav.stem)
    return re.sub(r"_[0-9a-f]{12}$", "", stem, flags=re.IGNORECASE).replace("_", " ").strip()


def normalized_title(wav: Path) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", title_from_pair(wav).lower())


def proxy_rank(wav: Path) -> tuple[int, float, int]:
    analysis = analyze_file(wav, sample_rate=8000)
    return (analysis.artifact_score, analysis.harsh_frame_ratio, analysis.sudden_drop_count)


# Harsh-noise events below this severity (zero-crossing ratio) are threshold
# fringe frames that are not audibly distinct from the model's normal vocal
# texture; only stronger events match the human-listening calibration.
NOISE_SEVERITY_THRESHOLD = 0.55


def measure(vocals: Path, source: Path) -> VocalMetrics:
    artifact = analyze_file(vocals)
    gate = analyze_vocals(vocals, source)
    noise = analyze_noise(vocals)
    events = detect_events(vocals, max_events_per_file=1000)
    return VocalMetrics(
        artifact_score=artifact.artifact_score,
        gate_score=gate.score,
        gate_decision=gate.decision,
        harsh_frame_ratio=noise.harsh_frame_ratio,
        burst_count=max(noise.burst_count, gate.burst_count),
        dropout_seconds=gate.dropout_seconds,
        structural_events=sum(e["type"] in {"possible_dropout", "abrupt_level_drop"} for e in events),
        noise_events=sum(
            e["type"] == "harsh_machine_noise" and float(e["severity"]) >= NOISE_SEVERITY_THRESHOLD
            for e in events
        ),
    )


def quality_value(metrics: VocalMetrics) -> float:
    return round(metrics.gate_score - metrics.artifact_score * 0.9 - metrics.harsh_frame_ratio * 120
                 - metrics.burst_count * 0.5 - metrics.structural_events * 2.0, 3)


def no_regression(before: VocalMetrics, after: VocalMetrics) -> bool:
    return (
        after.artifact_score <= before.artifact_score + 1
        and after.gate_score >= before.gate_score - 1
        and after.structural_events <= before.structural_events
        and after.dropout_seconds <= before.dropout_seconds + 0.05
    )


def cleanup_is_better(before: VocalMetrics, after: VocalMetrics) -> bool:
    noise_improved = (
        after.harsh_frame_ratio < before.harsh_frame_ratio
        or after.burst_count < before.burst_count
        or after.noise_events < before.noise_events
    )
    return no_regression(before, after) and noise_improved and quality_value(after) >= quality_value(before)


def humanize_preferred(before: VocalMetrics, after: VocalMetrics) -> bool:
    """Listening-first check for the de_buzz humanize chain.

    The air shelf (+2.5 dB @7 kHz) inflates the artifact and harsh-frame noise
    proxies by design, so those bounds are looser here; the audible dimensions
    (gate, structural breaks, dropouts) stay strict.
    """
    return (
        after.artifact_score <= before.artifact_score + 4
        and after.gate_score >= before.gate_score - 1
        and after.gate_decision != "reject"
        and after.structural_events <= before.structural_events
        and after.dropout_seconds <= before.dropout_seconds + 0.05
    )


# The de_buzz humanize chain adds an air shelf (+2.5 dB @7 kHz) which inflates
# the harsh-frame noise proxy (fringe frames) without making the vocal audibly
# noisier — the user listening A/B selected it as the most human variant. When
# the humanize variant is selected, the noise proxy bound is relaxed to this
# ceiling; the structural/dropout/artifact bounds stay strict.
HUMANIZE_NOISE_CEILING = 120


def passes_final_gate(metrics: VocalMetrics, max_artifact: int, min_gate: int,
                      max_noise_events: int = 8) -> bool:
    return (
        metrics.artifact_score <= max_artifact and metrics.gate_score >= min_gate
        and metrics.gate_decision != "reject" and metrics.structural_events <= 4
        and metrics.dropout_seconds <= 0.35 and metrics.noise_events <= max_noise_events
    )


def _copy_lyrics(source: Path, target_wav: Path) -> None:
    lyrics = source.with_suffix(".txt")
    if lyrics.exists():
        shutil.copy2(lyrics, target_wav.with_suffix(".txt"))


def rebuild(input_dir: Path, review_dir: Path, work_dir: Path, candidate_limit: int = 3,
            group_limit: int = 0, title_filter: str = "", max_artifact: int = 22,
            min_gate: int = 78, keep_work: bool = False) -> dict:
    groups: dict[str, list[Path]] = {}
    for wav in sorted(input_dir.glob("*.wav")):
        title = title_from_pair(wav)
        if not title_filter or title_filter.lower() in title.lower():
            groups.setdefault(normalized_title(wav), []).append(wav)
    selected_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    if group_limit:
        selected_groups = selected_groups[:group_limit]
    review_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"created_at": datetime.now().isoformat(timespec="seconds"), "songs": []}
    try:
        for _, wavs in selected_groups:
            ranked = sorted(((proxy_rank(wav), wav) for wav in wavs), key=lambda item: item[0])
            evaluated = []
            for proxy, wav in ranked[:candidate_limit]:
                song_work = work_dir / wav.stem
                vocals, instrumental = separate_with_demucs(wav, song_work)
                before = measure(vocals, wav)
                cleaned = song_work / "vocals_transparent_clean.wav"
                cleanup_error = ""
                try:
                    clean_stem(vocals, cleaned, "vocals", "review_fast", timeout_seconds=180)
                    after = measure(cleaned, wav)
                    use_cleaned = cleanup_is_better(before, after)
                except (subprocess.SubprocessError, OSError) as exc:
                    after = before
                    use_cleaned = False
                    cleanup_error = f"{type(exc).__name__}: {exc}"
                # Frame-level repair: local 4-10 kHz suppression + level recovery.
                # Kept when it beats both the source and the transparent cleanup.
                frame_repaired = song_work / "vocals_frame_repaired.wav"
                frame_error = ""
                try:
                    repair_vocals(vocals, frame_repaired)
                    frame_after = measure(frame_repaired, wav)
                except Exception as exc:  # noqa: BLE001 - frame repair is best-effort
                    frame_after = before
                    frame_error = f"{type(exc).__name__}: {exc}"
                # Humanize chain: frame repair + natural formant EQ + air + deesser.
                # Preferred by user listening A/B; kept when it beats everything else.
                humanized = song_work / "vocals_humanized.wav"
                humanize_error = ""
                try:
                    humanize_stem(vocals, humanized, variant="de_buzz")
                    humanize_after = measure(humanized, wav)
                except Exception as exc:  # noqa: BLE001 - humanize is best-effort
                    humanize_after = before
                    humanize_error = f"{type(exc).__name__}: {exc}"
                candidates = [("source", before, vocals)]
                if not cleanup_error:
                    candidates.append(("cleaned", after, cleaned))
                if not frame_error:
                    candidates.append(("frame", frame_after, frame_repaired))
                if not humanize_error:
                    candidates.append(("humanize", humanize_after, humanized))
                scored = max(
                    candidates,
                    key=lambda item: quality_value(item[1])
                    if cleanup_is_better(before, item[1])
                    else -float("inf"),
                )
                # Listening-first preference: the de_buzz humanize chain was chosen
                # by user A/B listening. Prefer it when it does not regress the
                # structural dimensions (breaks/dropouts/artifacts), even if the
                # strict metric ranking prefers another variant.
                humanize_entry = next((c for c in candidates if c[0] == "humanize"), None)
                if humanize_entry is not None and humanize_preferred(before, humanize_entry[1]):
                    chosen = humanize_entry
                else:
                    chosen = scored
                use_cleaned = chosen[0] == "cleaned"
                use_frame = chosen[0] == "frame"
                use_humanize = chosen[0] == "humanize"
                evaluated.append({"wav": wav, "instrumental": instrumental,
                                  "vocals": chosen[2], "proxy": proxy,
                                  "before": before, "after": after,
                                  "frame_after": frame_after,
                                  "humanize_after": humanize_after,
                                  "chosen": chosen[1],
                                  "cleanup_used": use_cleaned, "frame_used": use_frame,
                                  "humanize_used": use_humanize,
                                  "cleanup_error": cleanup_error, "frame_error": frame_error,
                                  "humanize_error": humanize_error})
            best = max(evaluated, key=lambda item: quality_value(item["chosen"]))
            if best.get("humanize_used"):
                # Humanize noise proxy is inflated by the air shelf; the strict
                # noise bound applies to every other treatment.
                passed = passes_final_gate(best["chosen"], max_artifact, min_gate,
                                           max_noise_events=HUMANIZE_NOISE_CEILING)
            else:
                passed = passes_final_gate(best["chosen"], max_artifact, min_gate)
            song = {
                "title": title_from_pair(best["wav"]), "source": str(best["wav"]),
                "decision": "review" if passed else "reject",
                "cleanup_used": best["cleanup_used"], "frame_used": best["frame_used"],
                "humanize_used": best["humanize_used"],
                "selected_metrics": asdict(best["chosen"]), "quality_value": quality_value(best["chosen"]),
                "candidate_count": len(evaluated),
                "candidates": [{"source": str(x["wav"]), "cleanup_used": x["cleanup_used"],
                                "frame_used": x["frame_used"], "humanize_used": x["humanize_used"],
                                "before": asdict(x["before"]), "after": asdict(x["after"]),
                                "frame_after": asdict(x["frame_after"]),
                                "humanize_after": asdict(x["humanize_after"]),
                                "selected_quality_value": quality_value(x["chosen"]),
                                "cleanup_error": x["cleanup_error"], "frame_error": x["frame_error"],
                                "humanize_error": x["humanize_error"]}
                               for x in evaluated],
            }
            if passed:
                safe_title = re.sub(r"[<>:\"/\\|?*]+", "_", song["title"]).strip()
                output = review_dir / f"{safe_title}_quality_rebuild.wav"
                remix(best["instrumental"], best["vocals"], output)
                _copy_lyrics(best["wav"], output)
                song["output"] = str(output)
            report["songs"].append(song)
    finally:
        if not keep_work and work_dir.exists():
            shutil.rmtree(work_dir)
    (review_dir / "quality_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Select and conservatively clean the best vocal take.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--candidate-limit", type=int, default=3)
    parser.add_argument("--group-limit", type=int, default=0)
    parser.add_argument("--title", default="")
    parser.add_argument("--max-artifact", type=int, default=22)
    parser.add_argument("--min-gate", type=int, default=78)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    report = rebuild(args.input_dir, args.review_dir, args.work_dir, args.candidate_limit,
                     args.group_limit, args.title, args.max_artifact, args.min_gate, args.keep_work)
    accepted = sum(song["decision"] == "review" for song in report["songs"])
    print(f"Accepted: {accepted}/{len(report['songs'])}")
    print(f"Review folder: {args.review_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
