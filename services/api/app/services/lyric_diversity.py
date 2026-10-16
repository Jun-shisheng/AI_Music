"""Lyric diversity / anti-fatigue checks for arranged songs.

Allows purposeful chorus hooks while rejecting mechanical verse reuse
and wall-to-wall identical lines that fatigue listeners.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SECTION_RE = re.compile(r"\[([^\]]+)\]")


@dataclass
class LyricDiversityReport:
    passes: bool
    unique_ratio: float
    line_count: int
    unique_line_count: int
    max_consecutive_repeats: int
    verse_similarity: float
    issues: list[str]

    def summary(self) -> str:
        if self.passes:
            return "lyric diversity ok"
        return "; ".join(self.issues) if self.issues else "lyric diversity failed"


def _clean_lines(lyrics: str) -> list[str]:
    lines: list[str] = []
    for line in lyrics.splitlines():
        cleaned = SECTION_RE.sub("", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _section_blocks(lyrics: str) -> dict[str, list[str]]:
    """Map normalized section labels to lyric lines under that section."""
    blocks: dict[str, list[str]] = {}
    current = "_preamble"
    for raw in lyrics.splitlines():
        tag = SECTION_RE.search(raw.strip())
        if tag and raw.strip().startswith("["):
            current = tag.group(1).strip().lower()
            blocks.setdefault(current, [])
            continue
        cleaned = SECTION_RE.sub("", raw).strip()
        if cleaned:
            blocks.setdefault(current, []).append(cleaned)
    return blocks


def _is_chorus_label(label: str) -> bool:
    return "副歌" in label or "chorus" in label


def _is_verse_label(label: str) -> bool:
    return "主歌" in label or label.startswith("verse")


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _max_consecutive_repeats(lines: list[str]) -> int:
    if not lines:
        return 0
    best = 1
    run = 1
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _ngram_fatigue(lines: list[str], n: int = 3) -> float:
    """Fraction of n-grams that appear more than once (0=fresh, 1=all reused)."""
    if len(lines) < n:
        return 0.0
    grams = ["||".join(lines[i : i + n]) for i in range(len(lines) - n + 1)]
    if not grams:
        return 0.0
    counts: dict[str, int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    reused = sum(1 for c in counts.values() if c > 1)
    return reused / len(counts)


def analyze_lyrics(lyrics: str) -> LyricDiversityReport:
    lines = _clean_lines(lyrics)
    issues: list[str] = []
    line_count = len(lines)
    unique_line_count = len(set(lines))
    unique_ratio = unique_line_count / line_count if line_count else 1.0
    max_consec = _max_consecutive_repeats(lines)
    blocks = _section_blocks(lyrics)

    verses = [lines_ for label, lines_ in blocks.items() if _is_verse_label(label) and lines_]
    verse_sim = 0.0
    if len(verses) >= 2:
        verse_sim = max(_jaccard(verses[i], verses[j]) for i in range(len(verses)) for j in range(i + 1, len(verses)))

    # Chorus lines may repeat; exclude pure chorus-only duplicates from unique_ratio penalty soft band.
    non_chorus_lines: list[str] = []
    for label, blines in blocks.items():
        if _is_chorus_label(label):
            continue
        non_chorus_lines.extend(blines)
    non_chorus_unique = (
        len(set(non_chorus_lines)) / len(non_chorus_lines) if non_chorus_lines else 1.0
    )

    if line_count >= 8 and unique_ratio < 0.55:
        issues.append(f"全曲重复过高 unique_ratio={unique_ratio:.2f}（需≥0.55）")
    if line_count >= 8 and non_chorus_unique < 0.75:
        issues.append(f"非副歌段落重复过高 unique={non_chorus_unique:.2f}（需≥0.75）")
    if max_consec >= 3:
        issues.append(f"连续相同歌词行过多（{max_consec} 行）")
    if verse_sim >= 0.6:
        issues.append(f"主歌之间过于相似 similarity={verse_sim:.2f}（需<0.60）")

    fatigue = _ngram_fatigue(lines, n=3)
    if line_count >= 12 and fatigue >= 0.35:
        issues.append(f"三连句复用偏多 ngram_fatigue={fatigue:.2f}（需<0.35）")

    # Require verse2-like progress when multiple verses exist: at least one unique line in later verse
    if len(verses) >= 2:
        first = set(verses[0])
        later_unique = any(line not in first for verse in verses[1:] for line in verse)
        if not later_unique:
            issues.append("后续主歌没有相对主歌1的新词句，叙事未推进")

    return LyricDiversityReport(
        passes=not issues,
        unique_ratio=round(unique_ratio, 4),
        line_count=line_count,
        unique_line_count=unique_line_count,
        max_consecutive_repeats=max_consec,
        verse_similarity=round(verse_sim, 4),
        issues=issues,
    )


def assert_lyric_diversity(lyrics: str) -> LyricDiversityReport:
    report = analyze_lyrics(lyrics)
    if not report.passes:
        raise ValueError(f"Lyric diversity check failed: {report.summary()}")
    return report


# --- strict repetition audit (2026-10-15) -----------------------------------
#
# The user's rule for battle songs, stated after listening to the 2026-10-14
# batch: "歌词可以重复但不能大片重复，开头高潮和结尾部分的歌词最好就是全新的
# 不重复，中间可以有重复但重复两遍顶天了". analyze_lyrics() above tolerates
# chorus repetition by design (it excludes chorus lines from the unique-ratio
# penalty), which is exactly what the user is now rejecting: a chorus hook sung
# six times across three choruses reads as padding.
#
# This is the stricter, separately-callable audit: no line -- hook included --
# may appear more than `max_repeats` times anywhere in the song, and the intro,
# the final chorus and the outro must be text that has not been sung before.

_NORM_RE = re.compile(r"[\s，。、！？；：,.!?;:\"'“”‘’()（）\-—~～…]+")


def _norm_line(line: str) -> str:
    return _NORM_RE.sub("", line).lower()


def ordered_sections(lyrics: str) -> list[tuple[str, list[str]]]:
    """Sections in song order, preserving repeated labels (two choruses = two entries)."""
    sections: list[tuple[str, list[str]]] = []
    for raw in lyrics.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        tag = SECTION_RE.fullmatch(stripped)
        if tag:
            sections.append((tag.group(1).strip().lower(), []))
            continue
        cleaned = SECTION_RE.sub("", stripped).strip()
        if cleaned:
            if not sections:
                sections.append(("_preamble", []))
            sections[-1][1].append(cleaned)
    return sections


def _is_final_chorus(label: str) -> bool:
    return ("final" in label or "终" in label or "最后" in label) and _is_chorus_label(label)


def _is_outro(label: str) -> bool:
    return "outro" in label or "尾奏" in label


def _is_intro(label: str) -> bool:
    return "intro" in label or "前奏" in label


@dataclass
class RepetitionReport:
    passes: bool
    max_repeat: int
    repeated: dict[str, int]
    stale_sections: list[str]
    issues: list[str]

    def summary(self) -> str:
        if self.passes:
            return "repetition ok"
        return "; ".join(self.issues) if self.issues else "repetition failed"


def audit_repetition(lyrics: str, max_repeats: int = 2) -> RepetitionReport:
    """Reject mass repetition and recycled opening/climax/ending lines.

    `max_repeats` is the largest number of times any single line may be sung
    (default 2 = "重复两遍顶天了").
    """
    sections = ordered_sections(lyrics)
    lines = [_norm_line(ln) for _, block in sections for ln in block]
    lines = [ln for ln in lines if ln]
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    over = {ln: c for ln, c in counts.items() if c > max_repeats and len(ln) >= 2}

    issues: list[str] = []
    if over:
        shown = ", ".join(f"\"{ln[:40]}\"x{c}" for ln, c in sorted(
            over.items(), key=lambda kv: -kv[1])[:4])
        issues.append(f"{len(over)} 行重复超过 {max_repeats} 次：{shown}")

    # Freshness of the parts the user singled out. A line is "used" once it has
    # been sung in an earlier section.
    seen: set[str] = set()
    stale: list[str] = []
    for label, block in sections:
        if _is_intro(label) or _is_final_chorus(label) or _is_outro(label):
            if label == "_preamble":
                continue
            reused = [_norm_line(ln) for ln in block if _norm_line(ln) in seen]
            if reused:
                stale.append(label)
                issues.append(
                    f"[{label}] 有 {len(reused)} 行与前文重复（开头/高潮/结尾需全新）："
                    + "; ".join(f"\"{r[:30]}\"" for r in reused[:3]))
        seen.update(_norm_line(ln) for ln in block if _norm_line(ln))

    return RepetitionReport(
        passes=not issues,
        max_repeat=max(counts.values()) if counts else 0,
        repeated={ln: c for ln, c in counts.items() if c > 1},
        stale_sections=stale,
        issues=issues,
    )
