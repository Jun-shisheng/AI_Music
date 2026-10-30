from __future__ import annotations

import re


SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
CJK_RE = re.compile(r"[\u3400-\u9fff]")
WORDLESS_RE = re.compile(
    r"^\s*(?:[啊呀哈啦呜喔哦嗨哼嗯]+|(?:ah+|oh+|o+h+|ooh+|woo+|la+|na+|mm+|hmm+)[\s,\-.!]*)+\s*$",
    re.IGNORECASE,
)
WORDLESS_CHARS_RE = re.compile(r"^[啊呀哈啦呜喔哦嗨哼嗯]+$")
WORDLESS_LATIN_RE = re.compile(r"^(?:ah+|oh+|o+h+|ooh+|woo+|la+|na+|mm+|hmm+)+$", re.IGNORECASE)

INTRO_LABELS = {"intro", "前奏", "开场", "引子"}
INTERLUDE_LABELS = {"interlude", "间奏"}


def detect_vocal_language(lyrics: str, style_prompt: str = "") -> str:
    text = f"{lyrics}\n{style_prompt}"
    if CJK_RE.search(text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"


def is_wordless_filler(line: str) -> bool:
    if WORDLESS_RE.match(line):
        return True
    compact = re.sub(r"[\s,\-.!，。！？、~]+", "", line).strip()
    if not compact:
        return False
    return bool(WORDLESS_CHARS_RE.match(compact) or WORDLESS_LATIN_RE.match(compact))


def prepare_generation_lyrics(lyrics: str) -> str:
    """Normalize lyrics to reduce unwanted humming and wordless AI vocal filler."""
    if not lyrics.strip():
        return lyrics

    prepared: list[str] = []
    current_section = ""
    skipping_intro_text = False

    for raw_line in lyrics.splitlines():
        line = raw_line.rstrip()
        match = SECTION_RE.match(line)
        if match:
            label = match.group(1).strip()
            normalized = label.lower()
            current_section = normalized
            skipping_intro_text = normalized in INTRO_LABELS
            if skipping_intro_text:
                prepared.append("[Instrumental Intro]")
            elif normalized in INTERLUDE_LABELS:
                prepared.append("[Instrumental Interlude]")
            else:
                prepared.append(line)
            continue

        if skipping_intro_text:
            if not line.strip():
                continue
            continue

        if current_section in INTERLUDE_LABELS and is_wordless_filler(line):
            continue
        if is_wordless_filler(line):
            continue
        prepared.append(line)

    cleaned = "\n".join(prepared)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or lyrics
