"""Reference-informed tag stacks for the 2026-10-29 film-score batch.

Why this exists
---------------
The 2026-10-28 batch measured against three film-score references that the
user named (《凡人修仙传》/《灵笼》):

    ref                     width   sub dB  centroid  dyn dB
    凡人 快哉风              0.496   32.2    1710      13.3
    灵笼 Incarnation         0.509   32.4    1991      19.4
    灵笼 佩妮的日记           0.500   25.8    1221      20.7
    10-28 我们 (10 首)       0.234-0.462  26.8-31.5  1570-2530  11.7-21.1

The consistent gap is stereo width (~0.33 vs ~0.50) and low end
(-3..-4 dB sub); a few of ours are also 400-600 Hz brighter than the
references.  The old tag heads asked for "wide cinematic stereo image" once,
buried in a pop-shaped tag list.  These stacks repeat the width/low-end
request in several forms and add the two palettes the references actually
use: guofeng orchestra (dizi/xiao/erhu/guzheng/pipa, taiko, low strings) and
sci-fi ritual (industrial sub hits, tribal drums, dark synth, distant choir).

Usage:
  python tools/generation/film_tags_20261029.py \
      --inputs inputs/2026-10-29_film_ref
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Shared tail: what made the 10-28 tracks read as "pop mix even when the
# arrangement was orchestral".
TAIL = [
    "cinematic production", "wide dynamic range", "deep low end",
    "expansive stereo image", "orchestral depth", "natural hall reverb",
    "clear human vocal", "natural breath", "precise consonants",
    "expressive legato", "controlled vibrato only at phrase ends",
    "no pop production", "no acoustic guitar", "no drum machine",
    "no edm drop", "no trap beat", "no sweet soft tone", "no vocoder",
    "no metallic resonance", "no synthetic humming", "no harsh digital noise",
    "no onomatopoeia", "no wordless vocalise", "no gang vocals",
    "no group chant",
]

# Film references sit around 0.50 stereo width.  State it three ways: the
# model's style encoder is a bag of phrases, not a parser.
WIDTH = [
    "wide cinematic stereo image",
    "hard-panned doubled strings and choir",
    "spacious stereo field",
    "warm analog mix, soft rolled-off highs",
]


def guofeng_film(seed: dict, extra: list[str]) -> list[str]:
    bpm, key = seed["bpm"], seed["key"]
    head = [
        "solo female vocal, warm chest-mix, clear diction, restrained and direct, "
        "lead vocal front and center",
        "cinematic Chinese fantasy film score, guofeng orchestral ballad",
        f"{bpm} bpm", key,
        "dizi flute", "xiao", "erhu", "guzheng", "pipa",
        "low strings drone, orchestral strings, french horn, timpani, taiko",
        "distant choir",
        "deep sub bass, low taiko",
        "sparse long phrases, quiet verses, solo dizi and strings in verses",
        "full orchestra enters in final chorus, slow build, soaring climax",
    ] + WIDTH + extra
    return head


def sci_fi_ritual(seed: dict, extra: list[str]) -> list[str]:
    bpm, key = seed["bpm"], seed["key"]
    head = [
        "solo female vocal, ethereal head voice with clear diction, present in the mix, "
        "long sustained notes, lead vocal front and center",
        "cinematic sci-fi ritual anthem, dark hybrid orchestral electronic",
        f"{bpm} bpm", key,
        "industrial sub bass hits, tribal war drums, taiko, dark synth pads, low brass",
        "distant choir, string ostinato, metallic percussion",
        "deep sub bass, sub drone",
        "huge hall reverb, future ritual atmosphere",
        "quiet verses, full ensemble in final chorus, slow build",
    ] + WIDTH + extra
    return head


def battle_film(seed: dict, extra: list[str]) -> list[str]:
    bpm, key = seed["bpm"], seed["key"]
    head = [
        "solo female lead vocal, powerful chest-mix belt, clear diction, "
        "front and center",
        "epic Chinese fantasy battle score, guofeng orchestral",
        f"{bpm} bpm", key,
        "taiko war drums, big drum, erhu, suona, dizi, guzheng",
        "low brass, war horns, strings ostinato, timpani",
        "distant male choir",
        "deep sub bass",
        "quiet verses, full ensemble in final chorus, driving ostinato",
        "cinematic climax",
    ] + WIDTH + extra
    return head


STACKS = {
    "guofeng_film": guofeng_film,
    "sci_fi_ritual": sci_fi_ritual,
    "battle_film": battle_film,
}

# Per-title extras: the one or two instruments that make each song not be
# the previous song.
EXTRAS = {
    "01_the_sword_returned": ["temple bell", "solo xiao in the intro"],
    "02_third_bell": ["low temple bell drone", "solo erhu in the bridge"],
    "03_names_worn_away": ["metallic wind hits", "solo cello in the verses"],
    "04_read_the_roll": ["tribal floor drums", "low male choir answers"],
    "05_the_pass_holds": ["suona cry", "war drum rolls"],
}

# Which palette each seed uses.  Modes stay the 10-28 ones so the lyric
# prompts (EPIC / EPIC_ETHEREAL / BATTLE) keep their register.
STACK_BY_SLUG = {
    "01_the_sword_returned": "guofeng_film",
    "02_third_bell": "guofeng_film",
    "03_names_worn_away": "sci_fi_ritual",
    "04_read_the_roll": "sci_fi_ritual",
    "05_the_pass_holds": "battle_film",
}


MODE_STACK = {
    "epic": "guofeng_film",
    "epic_ethereal": "sci_fi_ritual",
    "battle_anthem": "battle_film",
}


def build(seed: dict) -> str:
    slug = seed["slug"]
    stack = STACKS[STACK_BY_SLUG.get(slug) or MODE_STACK[seed["mode"]]]
    items = stack(seed, EXTRAS.get(slug, [])) + TAIL
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = " ".join(str(item).split())
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return ", ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path,
                    default=ROOT / "inputs" / "2026-10-29_film_ref")
    args = ap.parse_args()
    seeds = json.loads((args.inputs / "seeds.json").read_text(encoding="utf-8"))
    for seed in seeds:
        slug = seed["slug"]
        (args.inputs / f"{slug}_tags.txt").write_text(build(seed), encoding="utf-8")
        print(f"tags: {slug} <- {STACK_BY_SLUG.get(slug) or MODE_STACK[seed['mode']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
