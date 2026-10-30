"""Measure a YuE2 plan as a *melody*, not just as density.

`plan_metrics.py` answers "how many words per second and how high is the
register".  This tool answers the questions a songwriter would ask about the
tune itself, before anything is rendered:

  * is the chorus actually a lift, or just the verse louder?
  * is the melody mostly steps (singable) or a pile of leaps?
  * does the chorus return to the same hook, or does every chorus invent a new
    tune over the same chords?  (the 2026-10-16 finding: real choruses are the
    same hook again; ours were not)
  * where is the highest note, and does the last chorus go higher?
  * is the verse leaving room to breathe?

Everything is read from the symbolic ABC plan, so it costs milliseconds and can
gate candidate plans before a 6-20 minute render.  Thresholds are the temporary
Chinese-pop targets in `docs/lyric_melody_zh_2026.md` §3.3; they are diagnostic
until real Chinese reference profiles replace them.

Usage:
  P=vendor/ACE-Step-1.5/.venv/Scripts/python.exe
  $P -I tools/generation/melody_qa.py --file <score.abc> --target zh-pop
  $P -I tools/generation/melody_qa.py --dir outputs/... --json logs/melody.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_metrics import (  # noqa: E402
    _CHORD_RE,
    _TOKEN_RE,
    _length_units,
    plan_timing,
    voice_lines,
)
from yue2_plan_edit import _note_midi, key_accidentals, split_plan  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

TARGETS = {
    "zh-pop": {
        "density_ratio": (1.4, 4.5),
        "register_lift": (1.5, 8.0),
        "hook_similarity": (0.80, 1.0),
        "overall_range": (7, 20),
        "stepwise_share": (0.45, 1.0),
        "repeated_share": (0.0, 0.35),
        "climax_position": (0.55, 1.0),
        "verse_rest": (0.08, 0.45),
        "chorus_rest": (0.0, 0.28),
    },
    "diagnostic": {},
}


@dataclass
class Note:
    midi: int
    duration: float          # seconds
    start: float             # seconds from the start of the section


@dataclass
class SectionMelody:
    name: str
    seconds: float
    onsets: int
    density: float
    rest_share: float
    pitches: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)
    intervals: list[int] = field(default_factory=list)
    range_semitones: int = 0
    centre_midi: float = 0.0
    stepwise_share: float = 0.0
    leap_share: float = 0.0
    repeated_share: float = 0.0
    direction_changes: int = 0


@dataclass
class MelodyReport:
    passes: bool
    score: int
    target: str
    duration: float = 0.0
    sections: list[dict] = field(default_factory=list)
    verse_density: float = 0.0
    chorus_density: float = 0.0
    density_ratio: float = 0.0
    verse_centre: float = 0.0
    chorus_centre: float = 0.0
    register_lift: float = 0.0
    final_lift: float = 0.0
    hook_similarity: float | None = None
    overall_range: int = 0
    stepwise_share: float = 0.0
    leap_share: float = 0.0
    repeated_share: float = 0.0
    climax_position: float = 0.0
    climax_midi: int = 0
    longest_note_position: float = 0.0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.passes:
            return f"melody ok (score {self.score})"
        return f"melody FAILED (score {self.score}): " + "; ".join(self.issues[:4])


def _section_units(body: list[str]) -> float:
    lines = voice_lines(body, "Vocal") or voice_lines(body, None)
    total = 0.0
    for line in lines:
        clean = _CHORD_RE.sub("", line)
        for match in _TOKEN_RE.finditer(clean):
            if match.group(1):
                total += _length_units(match.group(2))
            else:
                total += _length_units(match.group(6))
    return total


def _notes(body: list[str], key: dict[str, int], sec_per_unit: float) -> list[Note]:
    lines = voice_lines(body, "Vocal")
    notes: list[Note] = []
    t = 0.0
    tied = False
    for line in lines:
        clean = _CHORD_RE.sub("", line)
        for match in _TOKEN_RE.finditer(clean):
            if match.group(1):                      # rest
                t += _length_units(match.group(2))
                tied = False
                continue
            units = _length_units(match.group(6))
            pitch = _note_midi(match.group(4), match.group(3) or "",
                               match.group(5), key)
            tie = bool(match.group(7))
            if tied and notes:
                notes[-1].duration += units * sec_per_unit
            else:
                notes.append(Note(pitch, units * sec_per_unit, t * sec_per_unit))
            t += units
            tied = tie
    return notes


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _histogram(pitches: list[int]) -> list[float]:
    hist = [0.0] * 12
    for p in pitches:
        hist[p % 12] += 1.0
    return hist


def _best_rotation_cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity after allowing any transposition.

    The final chorus is deliberately lifted by 2 semitones, so a raw
    pitch-class histogram would call the same hook a different hook.  A
    transposed hook is still the same hook; take the best of the 12 rotations.
    """
    best = 0.0
    for k in range(12):
        rotated = a[-k:] + a[:-k] if k else a
        best = max(best, _cosine(rotated, b))
    return best


def _hook_similarity(choruses: list[SectionMelody]) -> float | None:
    """How much the chorus sections sound like the same tune.

    Pitch-class histograms are octave-insensitive, which is exactly right: a
    final chorus may go up an octave and still be the same hook.  This is the
    symbolic analogue of `ref_profile.py`'s chroma hook_similarity, made
    transposition-invariant so `--final-lift 2` does not count as a new hook.
    """
    usable = [c for c in choruses if c.pitches]
    if len(usable) < 2:
        return None
    sims = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            sims.append(_best_rotation_cosine(_histogram(usable[i].pitches),
                                               _histogram(usable[j].pitches)))
    return round(sum(sims) / len(sims), 3)


def _section_melody(name: str, body: list[str], key: dict[str, int],
                    sec_per_unit: float) -> SectionMelody:
    sec = _section_units(body) * sec_per_unit
    notes = _notes(body, key, sec_per_unit)
    pitches = [n.midi for n in notes]
    intervals = [b - a for a, b in zip(pitches, pitches[1:])]
    steps = [i for i in intervals if i != 0]
    direction_changes = sum(
        1 for a, b in zip(steps, steps[1:]) if (a > 0) != (b > 0))
    return SectionMelody(
        name=name.lstrip("% ").strip(),
        seconds=round(sec, 2),
        onsets=len(pitches),
        density=round(len(pitches) / sec, 3) if sec > 0 else 0.0,
        rest_share=round(1 - sum(n.duration for n in notes) / sec, 3) if sec > 0 else 0.0,
        pitches=pitches,
        durations=[round(n.duration, 3) for n in notes],
        starts=[round(n.start, 3) for n in notes],
        intervals=intervals,
        range_semitones=(max(pitches) - min(pitches)) if pitches else 0,
        centre_midi=round(sum(pitches) / len(pitches), 1) if pitches else 0.0,
        stepwise_share=(round(sum(1 for i in intervals if 0 < abs(i) <= 2) / len(intervals), 3)
                        if intervals else 0.0),
        leap_share=(round(sum(1 for i in intervals if abs(i) >= 5) / len(intervals), 3)
                    if intervals else 0.0),
        repeated_share=(round(sum(1 for i in intervals if i == 0) / len(intervals), 3)
                        if intervals else 0.0),
        direction_changes=direction_changes,
    )


def _kind(label: str) -> str:
    low = label.lower()
    if "final chorus" in low:
        return "final_chorus"
    if "chorus" in low:
        return "chorus"
    if "verse" in low:
        return "verse"
    if "pre-chorus" in low or "prechorus" in low:
        return "pre_chorus"
    if "bridge" in low:
        return "bridge"
    if "intro" in low:
        return "intro"
    if "outro" in low:
        return "outro"
    return "other"


def audit(abc: str, target: str = "zh-pop") -> MelodyReport:
    header, raw_sections = split_plan(abc)
    key_name = next((ln[2:].strip() for ln in header if ln.startswith("K:")), "C")
    key = key_accidentals(key_name)
    sec_per_unit, _ = plan_timing(abc)

    sections = [_section_melody(name, body, key, sec_per_unit)
                for name, body in raw_sections]
    total = sum(s.seconds for s in sections)
    timed = []
    cursor = 0.0
    for s in sections:
        timed.append((cursor, s))
        cursor += s.seconds

    by_kind: dict[str, list[SectionMelody]] = {}
    for s in sections:
        by_kind.setdefault(_kind(s.name), []).append(s)

    def mean_density(kind: str) -> float:
        group = [s.density for s in by_kind.get(kind, []) if s.onsets]
        return round(sum(group) / len(group), 3) if group else 0.0

    def mean_centre(kind: str) -> float:
        group = [s.centre_midi for s in by_kind.get(kind, []) if s.pitches]
        return round(sum(group) / len(group), 1) if group else 0.0

    verse_density = mean_density("verse")
    chorus_density = mean_density("chorus")
    density_ratio = round(chorus_density / verse_density, 2) if verse_density else 0.0
    verse_centre = mean_centre("verse")
    chorus_centre = mean_centre("chorus")
    register_lift = round(chorus_centre - verse_centre, 1) if verse_centre and chorus_centre else 0.0
    final_centre = mean_centre("final_chorus")
    final_lift = round(final_centre - chorus_centre, 1) if final_centre and chorus_centre else 0.0

    choruses = by_kind.get("chorus", []) + by_kind.get("final_chorus", [])
    hook_sim = _hook_similarity(choruses)

    all_pitches = [p for s in sections for p in s.pitches]
    all_intervals = [i for s in sections for i in s.intervals]
    overall_range = (max(all_pitches) - min(all_pitches)) if all_pitches else 0
    stepwise = (round(sum(1 for i in all_intervals if 0 < abs(i) <= 2) / len(all_intervals), 3)
                if all_intervals else 0.0)
    leap = (round(sum(1 for i in all_intervals if abs(i) >= 5) / len(all_intervals), 3)
            if all_intervals else 0.0)
    repeated = (round(sum(1 for i in all_intervals if i == 0) / len(all_intervals), 3)
                if all_intervals else 0.0)

    climax_midi = max(all_pitches) if all_pitches else 0
    climax_position = 0.0
    longest_position = 0.0
    for start, s in timed:
        for note in zip(s.pitches, s.starts, s.durations):
            midi, rel, _ = note
            if midi == climax_midi and climax_position == 0.0 and total:
                climax_position = round((start + rel) / total, 3)
        if s.durations and total:
            idx = max(range(len(s.durations)), key=lambda i: s.durations[i])
            if s.durations[idx] >= max((max(x.durations) for x in sections if x.durations),
                                       default=0):
                longest_position = round((start + s.starts[idx]) / total, 3)

    report = MelodyReport(
        passes=False, score=100, target=target, duration=round(total, 2),
        sections=[asdict(s) for s in sections],
        verse_density=verse_density, chorus_density=chorus_density,
        density_ratio=density_ratio, verse_centre=verse_centre,
        chorus_centre=chorus_centre, register_lift=register_lift,
        final_lift=final_lift, hook_similarity=hook_sim,
        overall_range=overall_range, stepwise_share=stepwise,
        leap_share=leap, repeated_share=repeated,
        climax_position=climax_position, climax_midi=climax_midi,
        longest_note_position=longest_position,
    )

    # --- thresholds ------------------------------------------------------
    t = TARGETS.get(target, {})
    critical = False
    if not by_kind.get("verse") or not by_kind.get("chorus"):
        report.issues.append("计划缺少主歌或副歌的人声")
        critical = True
    if t:
        lo, hi = t["density_ratio"]
        if verse_density and chorus_density and not lo <= density_ratio <= hi:
            if density_ratio < lo:
                report.issues.append(
                    f"副歌/主歌密度比 {density_ratio}（需≥{lo}）：副歌不够密")
                critical = True
            else:
                report.warnings.append(
                    f"副歌/主歌密度比 {density_ratio}（>{hi}）：副歌可能过密/像喊")
        lo, hi = t["register_lift"]
        if verse_centre and chorus_centre and not lo <= register_lift <= hi:
            if register_lift < lo:
                report.issues.append(
                    f"副歌音区只抬升 {register_lift} 半音（需≥{lo}）")
                critical = True
            else:
                report.warnings.append(
                    f"副歌音区抬升 {register_lift} 半音（>{hi}）：可能超出舒适区")
        if hook_sim is not None:
            lo, hi = t["hook_similarity"]
            if hook_sim < lo:
                report.issues.append(
                    f"副歌钩子相似度 {hook_sim}（需≥{lo}）：三段副歌不是同一个 hook")
                critical = critical or hook_sim < 0.6
        lo, hi = t["overall_range"]
        if overall_range > hi:
            report.issues.append(
                f"全曲音域 {overall_range} 半音（>{hi}）：超过 1.5 个八度")
        lo, hi = t["stepwise_share"]
        if stepwise < lo:
            # Diagnostic, not a veto: the user's complaint is "not enough pitch
            # movement", which is the opposite of a leapy melody.  A low
            # stepwise share is worth seeing, but it should not outrank a take
            # that has the density/register/hook shape right.
            report.warnings.append(
                f"级进比例 {stepwise:.0%}（<{lo:.0%}）：旋律偏跳，听得时候注意是否碎")
        lo, hi = t["repeated_share"]
        if repeated > hi:
            report.warnings.append(
                f"同音重复 {repeated:.0%}（>{hi:.0%}）：可能像念经")
        lo, hi = t["climax_position"]
        if climax_position and not lo <= climax_position <= hi:
            report.issues.append(
                f"最高音在 {climax_position:.0%} 处（应 {lo:.0%}-{hi:.0%}）：高潮位置不对")
        lo, hi = t["verse_rest"]
        verse_rests = [s.rest_share for s in by_kind.get("verse", [])]
        if verse_rests:
            verse_rest = sum(verse_rests) / len(verse_rests)
            if verse_rest < lo:
                report.warnings.append(
                    f"主歌留白 {verse_rest:.0%}（<{lo:.0%}）：主歌太满")
        lo, hi = t["chorus_rest"]
        chorus_rests = [s.rest_share for s in by_kind.get("chorus", [])]
        if chorus_rests:
            chorus_rest = sum(chorus_rests) / len(chorus_rests)
            if chorus_rest > hi:
                report.warnings.append(
                    f"副歌留白 {chorus_rest:.0%}（>{hi:.0%}）：副歌太空")
    if final_lift < 0:
        report.warnings.append(f"末副歌比首副歌低 {abs(final_lift)} 半音")

    # --- score -----------------------------------------------------------
    score = 100
    score -= 15 * len([i for i in report.issues if "密度" in i or "音区" in i])
    score -= 12 * len([i for i in report.issues if "钩子" in i])
    score -= 8 * len([i for i in report.issues if "音域" in i
                      or "高潮" in i or "人声" in i])
    score -= min(10, 2 * len(report.warnings))
    report.score = max(0, min(100, score))
    report.passes = (not critical) and report.score >= 75
    return report


def print_report(label: str, r: MelodyReport) -> None:
    print(f"\n== {label}  [{r.target}]  [{r.summary()}]")
    print(f"  时长 {r.duration:.1f}s / 音域 {r.overall_range} 半音"
          f" / 级进 {r.stepwise_share:.0%} / 跳进 {r.leap_share:.0%}"
          f" / 同音 {r.repeated_share:.0%}")
    print(f"  副歌/主歌 密度 {r.density_ratio}（{r.chorus_density} vs {r.verse_density}）"
          f" / 音区 +{r.register_lift} 半音 / 末副歌 {r.final_lift:+.1f}")
    if r.hook_similarity is not None:
        print(f"  钩子相似度 {r.hook_similarity}")
    print(f"  最高音 {r.climax_midi} 在 {r.climax_position:.0%}"
          f" / 最长音在 {r.longest_note_position:.0%}")
    print(f"  {'section':<14}{'sec':>7}{'ons/s':>7}{'rest%':>7}"
          f"{'lo':>5}{'hi':>5}{'range':>7}{'step%':>7}{'rep%':>6}")
    for s in r.sections:
        pitches = s["pitches"]
        print(f"  {s['name']:<14}{s['seconds']:>7.1f}{s['density']:>7.2f}"
              f"{s['rest_share'] * 100:>7.0f}"
              f"{(min(pitches) if pitches else '-'):>5}"
              f"{(max(pitches) if pitches else '-'):>5}"
              f"{s['range_semitones']:>7}{s['stepwise_share'] * 100:>7.0f}"
              f"{s['repeated_share'] * 100:>6.0f}")
    for issue in r.issues:
        print(f"  ! {issue}")
    for warning in r.warnings:
        print(f"  ~ {warning}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="*", type=Path)
    ap.add_argument("--file", type=Path, action="append", default=[])
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--target", choices=sorted(TARGETS), default="zh-pop")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    paths = list(args.plans) + list(args.file)
    if args.dir:
        paths += sorted(args.dir.rglob("score.abc"))
    if not paths:
        ap.error("give a plan file, --file, or --dir")

    payload = {}
    failed = False
    for path in paths:
        report = audit(path.read_text(encoding="utf-8-sig"), target=args.target)
        print_report(path.parent.name if path.name == "score.abc" else path.name, report)
        payload[str(path)] = asdict(report)
        failed = failed or not report.passes
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\njson -> {args.json}")
    if args.strict and failed:
        print("\nSTRICT: at least one plan failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
