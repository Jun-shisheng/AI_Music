"""Audit a Mandarin pop lyric before it is handed to YuE2.

Why this exists (2026-10-26): the 2026-10-22/23 Chinese batch was singable but
generic -- lines explained emotions instead of showing a scene, the chorus had
no standalone hook, and the usual AI register (星光/远方/梦想/世界/拥抱) filled
the space where a concrete object should be.  This tool turns the lyric-craft
rules in `.claude/skills/mandarin-pop-lyric/SKILL.md` into a report:

  * section structure and per-section line counts (the 2:30-3:30 budget),
  * rhyme: line-final pinyin finals grouped into Mandarin pop rhyme families,
    plus a check that line endings are content words, not 的/了/着,
  * tone: 3rd-tone runs, line-final 3rd tones, neutral-tone load, open-vowel
    line endings (the vowels that can carry a long/high note),
  * hook: the repeated chorus line, its length, how many times it returns,
  * concreteness: distinct physical objects and sensory words vs abstract
    register words,
  * cliche / AI-copywriter phrases (星河滚烫, 未来可期, 一路生花 ...),
  * repetition and verse progression, reusing the API service's
    `lyric_diversity` checks.

It is deliberately diagnostic-first: the score is a sorting aid, not a verdict
on whether the song is good.  `--strict` is for the generation gate (a bad
brief should never reach the GPU), not for human review.

Usage:
  P=vendor/ACE-Step-1.5/.venv/Scripts/python.exe
  $P -I tools/generation/zh_lyric_qa.py --file inputs/.../01_x_lyrics.txt
  $P -I tools/generation/zh_lyric_qa.py --dir inputs/2026-10-26_zh_pop --json logs/zh_qa.json
  $P -I tools/generation/zh_lyric_qa.py --file ... --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT / "services" / "api"))
from app.services.lyric_diversity import analyze_lyrics  # noqa: E402

try:
    from pypinyin import Style, lazy_pinyin
except ImportError:  # pragma: no cover - the ACE venv has it; clear message is better
    print("pypinyin is required: uv pip install --python vendor/ACE-Step-1.5/.venv/Scripts/python.exe pypinyin",
          file=sys.stderr)
    raise

CJK = re.compile(r"[㐀-鿿]")
SECTION_TAG = re.compile(r"^\s*\[([^\]]+)\]\s*$")
PUNCT = re.compile(r"[\s，。、！？；：,.!?;:\"'“”‘’()（）\-—~～…·]+")

EXPECTED_SECTIONS = [
    "intro", "verse 1", "pre-chorus 1", "chorus", "verse 2",
    "pre-chorus 2", "chorus 2", "bridge", "final chorus", "outro",
]

# Three modes, because 2026-10-26 the user named two very different references:
# 悬溺 (sparse, atmospheric, wordless hook) and 新·与天齐 (dense, slogan hook,
# guofeng battle).  One quota cannot police both.
MODES = {
    "narrative": {
        "line_range": (24, 32),
        "quotas": {
            "intro": (0, 2), "verse 1": (3, 6), "pre-chorus 1": (1, 4),
            "chorus": (3, 6), "verse 2": (3, 6), "pre-chorus 2": (1, 4),
            "chorus 2": (3, 6), "bridge": (1, 4), "final chorus": (3, 6),
            "outro": (0, 3),
        },
        "concrete_min": 4,
        "abstract_max": 4,
        "hook_len": (4, 10),
        "wordless_hook": False,
    },
    "atmospheric": {
        "line_range": (16, 24),
        "quotas": {
            "intro": (0, 2), "verse 1": (2, 5), "pre-chorus 1": (0, 3),
            "chorus": (2, 5), "verse 2": (2, 5), "pre-chorus 2": (0, 3),
            "chorus 2": (2, 5), "bridge": (0, 3), "final chorus": (2, 5),
            "outro": (0, 2),
        },
        "concrete_min": 3,
        "abstract_max": 6,
        "hook_len": (3, 10),
        # 2026-10-26 user: no onomatopoeia, no humming.  Borrow 悬溺's space and
        # temperature, not its wordless hook.
        "wordless_hook": False,
    },
    "guofeng": {
        "line_range": (22, 30),
        "quotas": {
            "intro": (0, 2), "verse 1": (3, 6), "pre-chorus 1": (1, 3),
            "chorus": (3, 6), "verse 2": (3, 6), "pre-chorus 2": (1, 3),
            "chorus 2": (3, 6), "bridge": (1, 3), "final chorus": (3, 6),
            "outro": (0, 2),
        },
        "concrete_min": 4,
        "abstract_max": 10,
        "hook_len": (5, 9),
        "wordless_hook": False,
    },
    "epic": {
        # 2026-10-26 late: the user redirected the week toward epic / BGM /
        # philosophical Mandarin songs (G.E.M.-style central metaphor), while
        # keeping the hook plain-spoken.  Quotas match the 3:00-3:30 window.
        "line_range": (24, 30),
        "quotas": {
            "intro": (0, 2), "verse 1": (3, 6), "pre-chorus 1": (1, 3),
            "chorus": (3, 6), "verse 2": (3, 6), "pre-chorus 2": (1, 3),
            "chorus 2": (3, 6), "bridge": (1, 4), "final chorus": (3, 6),
            "outro": (0, 2),
        },
        "concrete_min": 4,
        "abstract_max": 12,
        "hook_len": (5, 9),
        "wordless_hook": False,
    },
    "epic_rap": {
        # Battle epic with one rap verse: the rap needs 4-8 short lines, so the
        # verse-2 quota and the total line budget are wider than `epic`.
        "line_range": (24, 34),
        "quotas": {
            "intro": (0, 2), "verse 1": (3, 6), "pre-chorus 1": (1, 3),
            "chorus": (3, 6), "verse 2": (4, 8), "pre-chorus 2": (1, 3),
            "chorus 2": (3, 6), "bridge": (1, 4), "final chorus": (3, 6),
            "outro": (0, 2),
        },
        "concrete_min": 4,
        "abstract_max": 12,
        "hook_len": (5, 9),
        "wordless_hook": False,
    },
    "epic_ethereal": {
        # 2026-10-28: the user rejected the 10-26 batch as 糖水歌 and asked for
        # 空灵感 (诀爱-type: vast, floating, minor key, long lines, late
        # climax).  Fewer lines than `epic`: this line breathes.
        "line_range": (20, 28),
        "quotas": {
            "intro": (0, 2), "verse 1": (3, 5), "pre-chorus 1": (1, 3),
            "chorus": (3, 5), "verse 2": (3, 5), "pre-chorus 2": (1, 3),
            "chorus 2": (3, 5), "bridge": (1, 4), "final chorus": (3, 6),
            "outro": (0, 2),
        },
        "concrete_min": 4,
        "abstract_max": 14,
        "hook_len": (5, 10),
        "wordless_hook": False,
    },
    "battle_anthem": {
        # 2026-10-28: 新·与天齐-type.  Slogan hook on strong beats, concrete
        # battle props everywhere; the quota is the guofeng one with a wider
        # line budget so the hook can be shouted twice.
        "line_range": (22, 30),
        "quotas": {
            "intro": (0, 2), "verse 1": (3, 6), "pre-chorus 1": (1, 3),
            "chorus": (3, 6), "verse 2": (3, 6), "pre-chorus 2": (1, 3),
            "chorus 2": (3, 6), "bridge": (1, 3), "final chorus": (3, 6),
            "outro": (0, 2),
        },
        "concrete_min": 4,
        "abstract_max": 12,
        "hook_len": (5, 9),
        "wordless_hook": False,
    },
}
WORDLESS_LINE = re.compile(r"^[呀咿啊啦哦嗯哈呜喔嗨嘿哟]+$")

# Finals that a singer can open up on.  i/u/ü are deliberately absent: they
# close the mouth and are the wrong place for the highest, longest note.
OPEN_FINALS = {"a", "ia", "ua", "ai", "uai", "ao", "iao", "an", "ian", "uan",
               "van", "ang", "iang", "uang"}

RHYME_GROUPS = {
    "a": {"a", "ia", "ua"},
    "o": {"o", "uo"},
    "e": {"e", "ie", "ve"},
    "i": {"i"},
    "u": {"u"},
    "v": {"v", "vn"},
    "ai": {"ai", "uai"},
    "ei": {"ei", "ui", "uei"},
    "ao": {"ao", "iao"},
    "ou": {"ou", "iu", "iou"},
    "an": {"an", "ian", "uan", "van"},
    "en": {"en", "in", "un", "uen"},
    "ang": {"ang", "iang", "uang"},
    "eng": {"eng", "ing", "ong", "iong", "ueng"},
    "er": {"er"},
}

FUNCTION_ENDINGS = set("的了着吗呢吧啊呀哦嘛么")
ABSTRACT_WORDS = [
    "星光", "星河", "远方", "梦想", "光芒", "世界", "温柔", "拥抱", "眼泪",
    "孤独", "永恒", "翅膀", "繁花", "岁月", "时光", "心跳", "命运", "守护",
    "信仰", "天堂", "烟火", "彼岸", "天涯", "红尘", "执念", "宿命", "救赎",
    "破晓", "深渊", "烈焰", "誓言", "荣耀", "流浪", "坚强", "希望", "未来",
    "勇敢", "奇迹", "童话", "宇宙", "灵魂", "归宿", "远方", "山河", "众生",
]
AI_COPYWRITER = [
    "不负韶华", "不负", "奔赴山海", "奔赴", "山海", "万物", "治愈", "温柔以待",
    "闪闪发光", "一路生花", "未来可期", "心之所向", "素履以往", "眼里有光",
    "心向远方", "热爱可抵", "岁月漫长", "人间值得", "万事胜意", "星河滚烫",
    "乘风破浪", "披荆斩棘", "向阳而生", "逆风翻盘", "顶峰相见", "眼里有星河",
]
CONCRETE_NOUNS = [
    "门", "窗", "街", "路灯", "公交", "地铁", "末班车", "便利店", "雨伞",
    "外套", "围巾", "手机", "消息", "电梯", "楼道", "钥匙", "杯子", "碗",
    "筷子", "烟", "酒", "咖啡", "超市", "收银台", "塑料袋", "车票", "站台",
    "行李箱", "照片", "信", "纸条", "笔", "日历", "钟", "床", "枕头", "被子",
    "灯", "开关", "插座", "冰箱", "洗衣机", "阳台", "衣服", "鞋", "镜子",
    "口红", "耳机", "电影", "屏幕", "键盘", "雨", "雪", "风", "太阳", "月亮",
    "云", "树", "花", "猫", "狗", "鸟", "海", "河", "桥", "路", "楼", "房间",
    "厨房", "沙发", "桌子", "椅子", "地板", "墙", "门铃", "快递", "外卖",
    "账单", "工资", "简历", "工牌", "校服", "操场", "教室", "课本", "试卷",
    "铅笔", "橡皮", "篮球", "自行车", "摩托车", "出租车", "火车", "飞机",
    "机场", "车站", "医院", "药店", "公园", "长椅", "口罩", "围裙", "锅",
]
SENSORY_WORDS = [
    "咸", "甜", "苦", "辣", "冷", "热", "暖", "湿", "干", "香", "臭", "吵",
    "静", "亮", "暗", "疼", "痒", "麻", "重", "轻", "软", "硬", "粗糙",
    "光滑", "刺", "烫", "冰", "酸", "涩", "闷", "潮", "烫手", "冰凉",
]


@dataclass
class Section:
    label: str
    lines: list[str]

    @property
    def cjk_lines(self) -> list[str]:
        return [ln for ln in self.lines if CJK.search(ln)]


@dataclass
class LyricReport:
    passes: bool
    score: int
    mode: str = "narrative"
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sections: dict[str, int] = field(default_factory=dict)
    line_count: int = 0
    avg_line_chars: float = 0.0
    long_lines: list[str] = field(default_factory=list)
    hook: str = ""
    hook_count: int = 0
    hook_chars: int = 0
    hook_open_vowel: bool = False
    rhyme_main: str = ""
    rhyme_consistency: float = 0.0
    rhyme_by_group: dict[str, float] = field(default_factory=dict)
    rhyme_by_section: dict[str, float] = field(default_factory=dict)
    function_endings: list[str] = field(default_factory=list)
    open_vowel_endings: float = 0.0
    tone3_runs_max: int = 0
    tone3_line_endings: float = 0.0
    neutral_tone_ratio: float = 0.0
    concrete_hits: list[str] = field(default_factory=list)
    sensory_hits: list[str] = field(default_factory=list)
    abstract_hits: dict[str, int] = field(default_factory=dict)
    cliche_hits: dict[str, int] = field(default_factory=dict)
    repeat_max: int = 0
    unique_ratio: float = 0.0
    non_chorus_unique: float = 0.0
    section_errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.passes:
            return f"zh lyric ok (score {self.score})"
        return f"zh lyric FAILED (score {self.score}): " + "; ".join(self.issues[:4])


def parse_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        tag = SECTION_TAG.fullmatch(stripped)
        if tag:
            sections.append(Section(tag.group(1).strip().lower(), []))
            continue
        cleaned = PUNCT.sub("", stripped)
        if CJK.search(cleaned):
            if not sections:
                sections.append(Section("_preamble", []))
            sections[-1].lines.append(cleaned)
    return sections


def _finals(line: str) -> list[str]:
    return [f for f in lazy_pinyin(line, style=Style.FINALS, strict=False)
            if f and CJK.match(f) is None and f != line]


def _final_of_char(ch: str) -> str:
    values = lazy_pinyin(ch, style=Style.FINALS, strict=False)
    return values[0] if values else ""


def _tone_of_char(ch: str) -> int:
    values = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
    if not values:
        return 0
    match = re.search(r"([0-5])$", values[0])
    return int(match.group(1)) if match else 0


def _rhyme_group(final: str) -> str:
    for name, members in RHYME_GROUPS.items():
        if final in members:
            return name
    return final or "?"


def _line_final_char(line: str) -> str:
    chars = [c for c in line if CJK.match(c)]
    return chars[-1] if chars else ""


def _is_chorus(label: str) -> bool:
    """`pre-chorus` is not a chorus: its lines are not the hook."""
    return "chorus" in label and "pre" not in label


def _is_open_final(final: str) -> bool:
    return final in OPEN_FINALS


def _hook_from(chorus_lines: list[list[str]], min_len: int = 4,
               max_len: int = 10) -> tuple[str, int]:
    """The repeated line across chorus occurrences; longest repeat wins."""
    counts: dict[str, int] = {}
    for lines in chorus_lines:
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
    repeated = [(line, n) for line, n in counts.items() if n >= 2]
    if not repeated:
        return "", 0
    viable = [(ln, n) for ln, n in repeated if min_len <= len(ln) <= max_len]
    pool = viable or repeated
    return max(pool, key=lambda kv: (kv[1], len(kv[0])))


def _tone3_runs(line: str) -> int:
    tones = [_tone_of_char(c) for c in line if CJK.match(c)]
    best = run = 0
    for tone in tones:
        if tone == 3:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def audit(text: str, title: str = "", mode: str = "narrative") -> LyricReport:
    cfg = MODES[mode]
    sections = parse_sections(text)
    report = LyricReport(passes=False, score=100, mode=mode)
    labels = [s.label for s in sections if s.label != "_preamble"]
    counts = {s.label: len(s.cjk_lines) for s in sections if s.label != "_preamble"}
    report.sections = counts
    section_errors: list[str] = []

    # --- structure -------------------------------------------------------
    if labels != EXPECTED_SECTIONS:
        section_errors.append(f"段落标签/顺序不对：{labels}")
    for label, (lo, hi) in cfg["quotas"].items():
        n = counts.get(label)
        if n is None:
            section_errors.append(f"缺少 [{label}]")
        elif not lo <= n <= hi:
            section_errors.append(f"[{label}] {n} 行，应为 {lo}-{hi}")
    if section_errors:
        report.section_errors = section_errors
        report.issues.extend(section_errors[:3])

    lines = [ln for s in sections for ln in s.cjk_lines]
    report.line_count = len(lines)
    lo_lines, hi_lines = cfg["line_range"]
    if not lo_lines <= len(lines) <= hi_lines:
        report.issues.append(
            f"总句数 {len(lines)}，{mode} 模式应为 {lo_lines}-{hi_lines}")

    wordless_lines = [ln for ln in lines if WORDLESS_LINE.fullmatch(ln)]
    if wordless_lines:
        report.issues.append(
            f"出现拟声/哼唱句 {wordless_lines[:3]}（用户明确不要拟声词和哼唱）")

    lengths = [len(ln) for ln in lines]
    report.avg_line_chars = round(sum(lengths) / len(lengths), 1) if lengths else 0.0
    for line in lines:
        if len(line) > 14:
            report.long_lines.append(line)
        elif len(line) < 3:
            report.warnings.append(f"过短句：{line!r}")
    if report.long_lines:
        report.issues.append(f"{len(report.long_lines)} 句超过 14 字（一口气唱不完）")

    # --- rhyme -----------------------------------------------------------
    # Real songs rhyme *within* a section type, not across the whole lyric:
    # the verse may sit on -ou and the chorus on -ai, and that is correct.  So
    # measure each group (verse / pre / chorus / bridge) against its own main
    # rhyme, then weight by line count.
    rhyme_groups: list[str] = []
    section_rhymes: dict[str, list[str]] = {}
    for section in sections:
        for line in section.cjk_lines:
            final_char = _line_final_char(line)
            if not final_char:
                continue
            final = _final_of_char(final_char)
            group = _rhyme_group(final)
            rhyme_groups.append(group)
            label = section.label
            if "chorus" in label:
                key = "chorus"
            elif "verse" in label:
                key = "verse"
            elif "pre-chorus" in label:
                key = "pre"
            elif "bridge" in label:
                key = "bridge"
            else:
                key = label
            section_rhymes.setdefault(key, []).append(group)
            if final_char in FUNCTION_ENDINGS:
                report.function_endings.append(line)
            if _is_open_final(final):
                report.open_vowel_endings += 1
    if rhyme_groups:
        counts_r: dict[str, int] = {}
        for group in rhyme_groups:
            counts_r[group] = counts_r.get(group, 0) + 1
        report.rhyme_main = max(counts_r, key=counts_r.get)
        report.rhyme_by_group = {g: round(n / len(rhyme_groups), 2)
                                 for g, n in sorted(counts_r.items(), key=lambda kv: -kv[1])}
        weighted = 0.0
        total_lines = 0
        for key, groups in section_rhymes.items():
            if len(groups) < 2:
                continue
            own = max(set(groups), key=groups.count)
            ratio = groups.count(own) / len(groups)
            report.rhyme_by_section[key] = round(ratio, 2)
            weighted += ratio * len(groups)
            total_lines += len(groups)
        report.rhyme_consistency = round(weighted / total_lines, 2) if total_lines else 0.0
    report.open_vowel_endings = round(
        report.open_vowel_endings / len(rhyme_groups), 2) if rhyme_groups else 0.0
    if report.rhyme_consistency < 0.5:
        report.issues.append(
            f"段内主韵平均只占 {report.rhyme_consistency:.0%}（需≥50%）")
    if len(report.function_endings) > 1:
        report.issues.append(f"句尾落在虚词上：{len(report.function_endings)} 句")
    if report.open_vowel_endings < 0.35:
        report.warnings.append(
            f"开口音句尾只占 {report.open_vowel_endings:.0%}，长音/高音会难唱")

    # --- tone ------------------------------------------------------------
    all_chars = [c for line in lines for c in line if CJK.match(c)]
    tones = [_tone_of_char(c) for c in all_chars]
    report.neutral_tone_ratio = round(tones.count(0) / len(tones), 2) if tones else 0.0
    report.tone3_runs_max = max((_tone3_runs(line) for line in lines), default=0)
    if report.tone3_runs_max > 3:
        report.issues.append(f"一句内最多 {report.tone3_runs_max} 个三声连读（需≤3）")
    elif report.tone3_runs_max == 3:
        report.warnings.append("有一句出现 3 个三声连读，最好改成 2 个")
    line_end_tones = []
    for line in lines:
        final_char = _line_final_char(line)
        if final_char:
            line_end_tones.append(_tone_of_char(final_char))
    if line_end_tones:
        report.tone3_line_endings = round(line_end_tones.count(3) / len(line_end_tones), 2)
        if report.tone3_line_endings > 0.3:
            report.warnings.append(
                f"{report.tone3_line_endings:.0%} 的句尾是三声，长音容易倒字")
    if report.neutral_tone_ratio > 0.25:
        report.warnings.append(f"轻声占 {report.neutral_tone_ratio:.0%}，句子可能偏口语/松散")

    # --- hook ------------------------------------------------------------
    chorus_blocks = [s.cjk_lines for s in sections if _is_chorus(s.label)]
    report.hook, report.hook_count = _hook_from(
        chorus_blocks, min_len=cfg["hook_len"][0], max_len=cfg["hook_len"][1])
    wordless_hook = bool(report.hook and WORDLESS_LINE.fullmatch(report.hook))
    if not report.hook:
        report.issues.append("副歌没有重复出现的 hook 句")
    else:
        report.hook_chars = len(report.hook)
        hook_lo, hook_hi = cfg["hook_len"]
        if not hook_lo <= report.hook_chars <= hook_hi:
            report.issues.append(
                f"hook 长度 {report.hook_chars} 字，{mode} 模式应为 {hook_lo}-{hook_hi}")
        if wordless_hook and not cfg["wordless_hook"]:
            report.issues.append(f"{mode} 模式不允许无词 hook：{report.hook!r}")
        if report.hook_count > 3:
            report.issues.append(f"hook 出现 {report.hook_count} 次（需≤3）")
        elif report.hook_count < 2:
            report.issues.append("hook 只出现 1 次，副歌没有回归")
        hook_final = _final_of_char(_line_final_char(report.hook))
        report.hook_open_vowel = _is_open_final(hook_final)
        if not report.hook_open_vowel:
            report.warnings.append(
                f"hook 最后一个字不是开口音（{hook_final or '?'}），最高音会难唱")
        if not wordless_hook and any(word in report.hook for word in ABSTRACT_WORDS):
            report.warnings.append(f"hook 里有抽象大词：{report.hook!r}")
        if title:
            title_chars = set(title) & set(report.hook)
            if title and not title_chars:
                report.warnings.append(f"标题 {title!r} 与 hook 没有共享字")

    # --- concreteness / register ----------------------------------------
    joined = "\n".join(lines)
    report.concrete_hits = sorted({w for w in CONCRETE_NOUNS if w in joined})
    report.sensory_hits = sorted({w for w in SENSORY_WORDS if w in joined})
    report.abstract_hits = {w: joined.count(w) for w in ABSTRACT_WORDS if w in joined}
    report.cliche_hits = {w: joined.count(w) for w in AI_COPYWRITER if w in joined}
    if len(report.concrete_hits) < cfg["concrete_min"]:
        report.issues.append(
            f"具体物件只有 {len(report.concrete_hits)} 个"
            f"（{mode} 模式需≥{cfg['concrete_min']}）：{report.concrete_hits}")
    if len(report.sensory_hits) < 2:
        report.warnings.append(
            f"感官词只有 {len(report.sensory_hits)} 个（建议≥2）")
    abstract_total = sum(report.abstract_hits.values())
    if abstract_total > cfg["abstract_max"]:
        report.issues.append(
            f"抽象大词共 {abstract_total} 次（{mode} 模式需≤{cfg['abstract_max']}）")
    cliche_total = sum(report.cliche_hits.values())
    if cliche_total >= 1:
        report.issues.append(
            f"AI 文案腔命中 {cliche_total} 次：{report.cliche_hits}")
    if report.hook and any(w in report.hook for w in AI_COPYWRITER):
        report.issues.append("hook 用了 AI 文案腔短语")

    # --- repetition / progression (API service checks) -------------------
    diversity = analyze_lyrics(text)
    report.unique_ratio = diversity.unique_ratio
    non_chorus_lines = [ln for s in sections if not _is_chorus(s.label)
                        for ln in s.cjk_lines]
    report.non_chorus_unique = round(
        len(set(non_chorus_lines)) / max(1, len(non_chorus_lines)), 2)
    for issue in diversity.issues:
        report.issues.append(issue)
    line_counts: dict[str, int] = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1
    report.repeat_max = max(line_counts.values(), default=0)
    non_hook_over = {ln: n for ln, n in line_counts.items()
                     if n > 2 and ln != report.hook}
    if non_hook_over:
        report.issues.append(f"{len(non_hook_over)} 行（非 hook）重复超过 2 次")

    # --- score -----------------------------------------------------------
    critical = (bool(section_errors) or not report.hook or cliche_total >= 3
                or bool(wordless_lines))
    score = 100
    score -= 6 * len(section_errors)
    if not report.hook:
        score -= 20
    elif report.hook_count < 2 or report.hook_count > 3:
        score -= 8
    if report.rhyme_consistency < 0.5:
        score -= 10
    if len(report.function_endings) > 1:
        score -= 5
    if report.tone3_runs_max > 3:
        score -= 5
    if len(report.concrete_hits) < cfg["concrete_min"]:
        score -= 10
    if abstract_total > cfg["abstract_max"]:
        score -= 8
    if cliche_total:
        score -= min(20, 8 * cliche_total)
    if report.long_lines:
        score -= 5
    if non_hook_over:
        score -= 8
    if wordless_lines:
        score -= 15
    if not cfg["line_range"][0] <= report.line_count <= cfg["line_range"][1]:
        score -= 5
    report.score = max(0, min(100, score))
    report.passes = (not critical) and report.score >= 80
    if report.passes and report.warnings:
        report.score = max(80, report.score - min(5, len(report.warnings)))
    return report


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def print_report(label: str, report: LyricReport) -> None:
    print(f"\n== {label}  [{report.mode}]  [{report.summary()}]")
    print(f"  句数 {report.line_count} / 均长 {report.avg_line_chars} 字"
          f" / 主韵 {report.rhyme_main} 段内一致 {report.rhyme_consistency:.0%}"
          f" / 开口音句尾 {report.open_vowel_endings:.0%}")
    if report.rhyme_by_section:
        print(f"  分段韵脚一致度: {report.rhyme_by_section}")
    print(f"  hook {report.hook!r} x{report.hook_count}"
          f" / 开口音收尾 {'是' if report.hook_open_vowel else '否'}"
          f" / 三声连读 max {report.tone3_runs_max}"
          f" / 句尾三声 {report.tone3_line_endings:.0%}")
    print(f"  具体物件 {len(report.concrete_hits)}: {', '.join(report.concrete_hits[:12])}")
    if report.sensory_hits:
        print(f"  感官词: {', '.join(report.sensory_hits)}")
    if report.abstract_hits:
        print(f"  抽象词: {report.abstract_hits}")
    if report.cliche_hits:
        print(f"  AI 文案腔: {report.cliche_hits}")
    for issue in report.issues:
        print(f"  ! {issue}")
    for warning in report.warnings:
        print(f"  ~ {warning}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, action="append", default=[])
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--mode", choices=sorted(MODES), default="narrative")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    paths = list(args.file)
    if args.dir:
        paths += sorted(args.dir.glob("*_lyrics.txt"))
    if not paths:
        ap.error("give --file or --dir")

    payload = {}
    failed = False
    for path in paths:
        report = audit(_read_text(path), title=args.title, mode=args.mode)
        print_report(path.name, report)
        payload[str(path)] = asdict(report)
        failed = failed or not report.passes

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\njson -> {args.json}")
    if args.strict and failed:
        print("\nSTRICT: at least one lyric failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
