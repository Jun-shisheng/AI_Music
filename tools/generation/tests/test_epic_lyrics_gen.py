"""Tests for epic_lyrics_gen: the local gate on what the LLM wrote.

Everything here is offline -- the DeepSeek round trip is the untestable part,
so what is covered is the part that decides whether a brief is allowed onto
disk (structure tags, line budget, hook budget, repetition, English-only,
diversity) plus the tag-file builder and the .env reader.

The 2026-10-15 rules changed what "passes" means: the hook is now capped at two
occurrences, the intro / final chorus / outro must be previously unsung text,
and the arrangement tag file must spell out per-section dynamics.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epic_lyrics_gen import (  # noqa: E402
    SECTION_ORDER, TAG_ARC, _read_api_key, _build_user_prompt, build_tags,
    dedupe_segments, SEEDS, validate_brief,
)

SEED = SEEDS[0]

HOOK = "hold the dawn"


def _lyrics(verse1_lines: int = 3, verse2_lines: int = 3,
            final_chorus: str | None = None, hook_lines: int = 1) -> str:
    """A brief that satisfies every local rule, unless a knob says otherwise.

    `hook_lines` repeats the hook inside [Chorus 2] to test the two-occurrence
    cap (1 is legal, 2 makes three in the song).
    """
    body = ["[Intro]", "Cold bells over the wall", "Nothing left to burn"]
    body += ["", "[Verse 1]"]
    body += [f"Verse one line number {i} walks the frozen wall" for i in range(verse1_lines)]
    body += ["", "[Pre-Chorus 1]", "Ash on my tongue", "The cold climbs higher"]
    body += ["", "[Chorus]", f"{HOOK} until the morning", "Hold the gate, hold the grey",
             "Someone singing in the dark below", "I keep the watch I never chose"]
    body += ["", "[Verse 2]"]
    body += [f"Verse two line number {i} counts the names we lost" for i in range(verse2_lines)]
    body += ["", "[Pre-Chorus 2]", "My hands are stone", "The horn is rising now"]
    body += ["", "[Chorus 2]", f"{HOOK} until the morning", "The yard is silent, someone sings"]
    body += ["I have not slept since yesterday", "Every gate I hold is someone's name"]
    if hook_lines > 1:
        body[-1] = f"{HOOK} until the morning"
    body += ["", "[Bridge]", "The tower bell is cracked, it will not ring",
             "So I make the sound it cannot make"]
    body += ["", "[Final Chorus]"]
    body += (final_chorus.splitlines() if final_chorus else [
        "East is pale and I am standing",
        "Every name I kept is written on the wall",
        "The gate is mine, the grey is mine",
        "Dawn comes up like a breaking wave"])
    body += ["", "[Outro]", "Nobody left to count but me"]
    return "\n".join(body)


def _brief(**over) -> dict:
    data = {
        "title_en": "Cold Bells",
        "title_zh": "寒钟",
        "hook": HOOK,
        "lyrics": _lyrics(),
        "lyrics_zh": "[前奏]\n城墙上的寒钟……\n[主歌1]\n……",
        "style_extra": "solo cello ostinato, struck anvil percussion, distant "
                       "snow-field reverb, rising string ostinato, brass in the "
                       "final chorus",
    }
    data.update(over)
    return data


# --- structure / budget gate -------------------------------------------------

def test_well_formed_brief_passes():
    assert validate_brief(_brief(), SEED) == []


def test_wrong_section_tag_order_is_rejected():
    lyrics = _lyrics().replace("[Bridge]", "[Solo Break]")
    problems = validate_brief(_brief(lyrics=lyrics), SEED)
    assert any("section tags" in p for p in problems)


def test_overlong_brief_is_rejected():
    # The model overshot to 35-45 lines on its first pass; over 30 lines the
    # rendered song leaves the 2:00-3:30 window the user asked for.
    problems = validate_brief(_brief(lyrics=_lyrics(verse1_lines=8, verse2_lines=8)), SEED)
    assert any("sung lines" in p for p in problems)


def test_chinese_characters_in_lyrics_are_rejected():
    lyrics = _lyrics().replace("Cold bells over the wall", "Cold bells 寒钟")
    assert any("Chinese" in p for p in validate_brief(_brief(lyrics=lyrics), SEED))


def test_too_few_sung_lines_is_rejected():
    # Tags in the right order, but only one line under each -- a 10-line brief
    # would render far short of the 2:30 floor the user asked for.
    thin = "\n".join(f"[{tag}]\nshort line here" for tag in SECTION_ORDER)
    assert any("sung lines" in p for p in validate_brief(_brief(lyrics=thin), SEED))


def test_hook_must_appear_at_least_once():
    thin = _lyrics().replace(HOOK, "some other line")
    problems = validate_brief(_brief(lyrics=thin), SEED)
    assert any("hook" in p and "never appears" in p for p in problems)


def test_hook_never_sung_three_times():
    # The user: "重复两遍顶天了". A hook in [Chorus], [Chorus 2] and [Final
    # Chorus] was the 2026-10-14 shape and is exactly what is now rejected.
    problems = validate_brief(_brief(lyrics=_lyrics(hook_lines=2)), SEED)
    assert any("hook" in p and "max 2" in p for p in problems)


def test_recycled_final_chorus_is_rejected():
    # The climax must be new text, not the chorus sung again.
    recycled = "\n".join([f"{HOOK} until the morning", "Hold the gate, hold the grey",
                          "Someone singing in the dark below",
                          "I keep the watch I never chose"])
    problems = validate_brief(_brief(lyrics=_lyrics(final_chorus=recycled)), SEED)
    assert any("repetition" in p for p in problems), problems
    assert any("final chorus" in p for p in problems)


def test_outro_reusing_the_intro_is_rejected():
    lyrics = _lyrics().replace("[Outro]\nNobody left to count but me",
                               "[Outro]\nCold bells over the wall")
    problems = validate_brief(_brief(lyrics=lyrics), SEED)
    assert any("repetition" in p and "outro" in p for p in problems)


def test_a_line_may_not_be_sung_a_third_time():
    # "Ash on my tongue" sits in [Pre-Chorus 1]; smuggling it into [Verse 2] and
    # [Chorus 2] makes three, which is over the budget even away from the hook.
    lyrics = _lyrics()
    lyrics = lyrics.replace("Verse two line number 0 counts the names we lost",
                            "Ash on my tongue")
    lyrics = lyrics.replace("Someone singing in the dark below", "Ash on my tongue")
    problems = validate_brief(_brief(lyrics=lyrics), SEED)
    assert any("repetition" in p for p in problems), problems


def test_thin_style_extra_is_rejected():
    assert any("style_extra" in p
               for p in validate_brief(_brief(style_extra="loud drums"), SEED))


def test_missing_field_is_reported():
    data = _brief()
    del data["hook"]
    assert any("missing hook" in p for p in validate_brief(data, SEED))


# --- tag file ----------------------------------------------------------------

def test_build_tags_carries_voice_key_bpm_and_negatives():
    tags = build_tags(SEED, "solo cello ostinato, struck anvil percussion")
    assert not tags.endswith(",")
    assert SEED["voice"] in tags
    assert f"{SEED['bpm']} bpm" in tags and SEED["key"] in tags
    assert "solo cello ostinato" in tags
    assert "no male vocal" in tags
    assert "powerful female lead vocal singing in English" in tags


def test_build_tags_survives_multiline_style_extra():
    tags = build_tags(SEED, "  brass fanfare,\n\n taiko,\n ")
    assert "\n" not in tags
    assert "brass fanfare, taiko" in tags


def test_tags_demand_a_short_intro_and_real_dynamics():
    # 2026-10-15: the delivered tracks began singing at 15-32 s, and the user
    # called the dynamic range flat. Both are now stated in the caption.
    tags = build_tags(SEED, "solo cello, struck anvil")
    assert "vocal enters within the first two seconds" in tags
    assert "no instrumental intro" in tags
    assert "drums out" in tags and "the quietest point of the song" in tags
    assert "belted at the top of her range" in tags
    assert "extreme dynamic range" in tags
    assert "no flat even dynamics" in tags


def test_tags_carry_the_reference_devices():
    # Devices the reference songs use (HI3 "Starfall", Arknights "Radiant",
    # HSR "Wildfire" / "Flares of the Blazing Sun") -- see the TAG_ARC comment.
    tags = build_tags(SEED, "solo cello, struck anvil")
    assert "one to four seconds of total silence before the final chorus" in tags
    assert "modulates up a whole step" in tags
    assert "sub-bass drop on the downbeat" in tags
    assert "thirty percent of the chorus energy" in tags
    assert "no steady loud wall of sound" in tags


def test_tag_arc_states_every_section():
    for section in ("verse:", "pre-chorus:", "chorus:", "bridge:",
                    "final chorus:", "outro:"):
        assert section in TAG_ARC


def test_build_tags_dedupes_model_restated_tempo_and_key():
    tags = build_tags(SEED, f"{SEED['bpm']} bpm, {SEED['key']}, low cello drone")
    assert tags.lower().count(f"{SEED['bpm']} bpm") == 1
    assert tags.lower().count(SEED["key"].lower()) == 1
    assert "low cello drone" in tags


def test_dedupe_keeps_first_seen_order():
    assert dedupe_segments("a, b, a , c, B") == "a, b, c"


# --- prompt + env ------------------------------------------------------------

def test_user_prompt_lists_hooks_to_avoid():
    used = [{"title_en": "Cold Bells", "title_zh": "寒钟", "hook": HOOK,
             "first_line": "Cold bells over the wall"}]
    prompt = _build_user_prompt(SEED, used)
    assert HOOK in prompt and "Cold Bells" in prompt
    assert SEED["angle"] in prompt


def test_user_prompt_without_history_still_complete():
    prompt = _build_user_prompt(SEED, [])
    assert SEED["images"] in prompt and "Write this track now." in prompt


def test_env_reader(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("FOO=1\nDEEPSEEK_API_KEY=sk-test\n"
                   "DEEPSEEK_BASE_URL=https://example.invalid\n", encoding="utf-8")
    assert _read_api_key(env) == ("sk-test", "https://example.invalid")


def test_env_reader_defaults_base_url(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=sk-test\n", encoding="utf-8")
    assert _read_api_key(env) == ("sk-test", "https://api.deepseek.com")


def test_seeds_are_distinct_and_ascii():
    slugs = [s["slug"] for s in SEEDS]
    assert len(slugs) == len(set(slugs))
    assert all(slug.isascii() for slug in slugs)
    assert len({(s["key"], s["bpm"]) for s in SEEDS}) == len(SEEDS)


def test_stage_direction_inside_a_sung_line_is_rejected():
    # The 09 brief for the 2026-10-15 batch came back with
    # `I will wash the child's shoe tonight (held note on "night")`, and the
    # model sings what the line says -- so the note reaches the vocal and the
    # delivered sheet.
    lyrics = _lyrics().replace("East is pale and I am standing",
                               'East is pale (held note on "pale")')
    problems = validate_brief(_brief(lyrics=lyrics), SEED)
    assert any("stage direction" in p for p in problems)
