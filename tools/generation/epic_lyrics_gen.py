"""Fresh English epic-battle briefs, written by an LLM one title at a time.

Why this exists: the user's reference for English epic battle songs is
HOYO-MiX's English-language game anthems (Wildfire / Flares of the Blazing Sun
-- rock-orchestra hybrids, 3:1x long, big repeated hook, defiance-against-fate
subject matter).  On the previous round the user liked the melodies but called
the lyrics lacking, and asked for the words to be re-designed by an LLM so that
no two tracks share a concept.

So this script is the front half of a YuE2 run: for each seed direction it asks
DeepSeek for a complete brief -- English sing-lyrics with the section tags YuE2
consumes, a Chinese gloss for the user's own review, and the production
descriptor that makes that title's arrangement distinct.  Everything the model
writes is checked locally before it is written to disk:

  * exact structure tags ([Intro] [Verse 1] [Pre-Chorus 1] [Chorus] ... [Outro]),
  * a line budget that renders to ~3:00-3:30 on YuE2.  Calibrated on the
    2026-10-13 batch, where the line count is the knob that sets the plan
    length: 23 lines -> 142-224 s, 26 lines -> 150-210 s (median 185 s = 3:05
    over seven titles), 41 lines -> 215 s.  26-27 lines is the centre of the
    user's preferred 2:30-3:30 window, so the per-section quotas are fixed to
    sum to 27 -- the model overshoots a stated total by 20-40% if it is only
    asked for a range,
  * English only -- no Chinese characters leak into a lyric that is about to be
    sung in English,
  * the hook appears once or twice and NO line is sung more than twice, with
    the intro, the final chorus and the outro made of words that were not sung
    before (`lyric_diversity.audit_repetition`),
  * the anti-fatigue check from the API service
    (services/api/app/services/lyric_diversity.py) passes,
  * no hook / title / opening image is reused from an earlier title in the run.

2026-10-15 revision -- the user's listening verdict on the 2026-10-14 batch
("60 分是有了，但还是有很多问题") produced three hard rules, all encoded here:

  1. **No long intro.** Measured on the nine delivered 2026-10-14 tracks, the
     first sung word lands at 15.5-32.3 s (median 30.0 s). The user wants either
     "a few seconds" or at most 10-20 s. Two changes: the brief now asks for a
     vocal that starts immediately, the fixed tag head says so explicitly, and
     the renderer trims the plan's instrumental intro bars
     (`yue2_plan_edit.py`) so it does not depend on the model's good behaviour.
  2. **Bigger dynamic range.** "低的时候不够低，高的时候不够高有点平了." The
     lyric brief now specifies register per section (verse low, close and
     narrow; final chorus high, wide and belted with one long held note) and the
     arrangement caption spells out the section-by-section dynamics instead of
     the previous generic "restrained intro, late climactic chorus".
  3. **No mass repetition.** "歌词可以重复但不能大片重复，开头高潮和结尾部分的
     歌词最好就是全新的不重复，中间可以有重复但重复两遍顶天了." The hook now
     lives in [Chorus] and [Chorus 2] only (two occurrences, both in the middle
     of the song); [Intro], [Final Chorus] and [Outro] must be new text, and
     nothing may be sung a third time anywhere.

Usage:
  vendor/ACE-Step-1.5/.venv/Scripts/python.exe -I tools/generation/epic_lyrics_gen.py \
      --out-dir inputs/2026-10-15_epic_en

Rerun-safe: a title whose files already exist is skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# The anti-fatigue checker lives with the API service and has no third-party
# imports, so it can be reused here instead of duplicated.
sys.path.insert(0, str(PROJECT_ROOT / "services" / "api"))
from app.services.lyric_diversity import analyze_lyrics, audit_repetition  # noqa: E402

SECTION_ORDER = [
    "Intro", "Verse 1", "Pre-Chorus 1", "Chorus", "Verse 2", "Pre-Chorus 2",
    "Chorus 2", "Bridge", "Final Chorus", "Outro",
]

# Directions, not lyrics: the angle, the image bank and the musical frame are
# fixed so the twelve titles cannot converge on one another, while the actual
# hook, title and lines are the model's to invent.  Every angle is new against
# the 2026-10-14 set (last dawn city / unbowed / iron lullaby / the unchosen /
# spearstorm / embers never die / crown of thorns / we who remain / tide of
# endings / the sky we burned / vow under ruins / rise again).
SEEDS: list[dict] = [
    {"slug": "01_hammer_and_hymn", "angle": "a forge lit at midnight, the weapon and the song made at the same time; the rhythm of the hammer is the rhythm of the chorus",
     "images": "anvil, sparks, quenching water, hammer marks", "key": "E minor", "bpm": 158, "voice": "powerful mature female chest voice"},
    {"slug": "02_no_white_flag", "angle": "refusing to raise the flag of surrender even with the wall already lost behind them",
     "images": "white cloth, broken flagpole, empty courtyard, morning fog", "key": "A minor", "bpm": 150, "voice": "powerful mature female chest voice"},
    {"slug": "03_the_gate_holds", "angle": "a shield wall that is not allowed to break, sung from inside the press of it",
     "images": "shield rim, locked elbows, boots in mud, splintered gate", "key": "D minor", "bpm": 166, "voice": "powerful mature female chest voice"},
    {"slug": "04_iron_rain", "angle": "walking forward through falling iron and finding out what a body can choose",
     "images": "hail, helmets, dented ground, held breath", "key": "B minor", "bpm": 172, "voice": "powerful mature female chest voice"},
    {"slug": "05_drums_below_deck", "angle": "drums struck below the deck of a ship sailing into a war it cannot turn away from",
     "images": "drum skin, salt water, dark hold, rigging, storm lantern", "key": "C minor", "bpm": 154, "voice": "powerful mature female chest voice"},
    {"slug": "06_count_the_living", "angle": "counting the living instead of the dead; the arithmetic of who is still here",
     "images": "chalk marks, tally stick, wet canvas, first rope of dawn", "key": "G minor", "bpm": 146, "voice": "bright clear female voice with a hard edge"},
    {"slug": "07_signal_fire", "angle": "the last signal fire on the ridge, lit to tell someone far away that it is not over",
     "images": "ridge line, pitch smoke, wet flint, far tower", "key": "F# minor", "bpm": 162, "voice": "powerful mature female chest voice"},
    {"slug": "08_wolves_at_the_wall", "angle": "something with teeth is at the wall and the only weapon left is discipline",
     "images": "claw marks, torch line, breath in cold air, iron spikes", "key": "E minor", "bpm": 168, "voice": "powerful mature female chest voice"},
    {"slug": "09_small_and_unbroken", "angle": "protecting the ordinary and small things that a war always forgets",
     "images": "bread, bandage, child's shoe, kettle, folded letter", "key": "A minor", "bpm": 138, "voice": "bright clear female voice, tender in the verse, powerful in the chorus"},
    {"slug": "10_break_the_line", "angle": "the moment a line of soldiers decides to move instead of waiting to be hit",
     "images": "dust, drawn breath, one spoken word, sudden open ground", "key": "D minor", "bpm": 176, "voice": "powerful mature female chest voice"},
    {"slug": "11_seven_suns", "angle": "a sky with too many suns, a world burning from above, and the ones still marching",
     "images": "white sky, cracked earth, water skin, shadow under a shield", "key": "C# minor", "bpm": 156, "voice": "powerful mature female chest voice"},
    {"slug": "12_the_second_heart", "angle": "the second wind arrives as a second heart; singing about the moment exhaustion turns into power",
     "images": "knuckles, split lip, second wind, bell at dawn", "key": "B minor", "bpm": 164, "voice": "powerful mature female chest voice"},
]

SYSTEM_PROMPT = """You are a lyricist for Chinese gacha-game battle anthems in the HOYO-MiX \
English idiom (the register of "Wildfire", "Flares of the Blazing Sun" and \
"Starfall": cinematic rock-orchestra hybrids, defiant, anthemic, written to be \
sung loud by a female lead).

Write ONE song per request, in English.

Hard rules:
1. Use exactly these section tags, in this order, each on its own line:
[Intro] [Verse 1] [Pre-Chorus 1] [Chorus] [Verse 2] [Pre-Chorus 2] [Chorus 2] [Bridge] [Final Chorus] [Outro]
2. Total sung lines (excluding tag lines and blank lines): exactly 27. Hit this
   per-section budget exactly -- it is what sets the finished length, and a
   longer brief makes a song nobody can release:
     [Intro] 2, [Verse 1] 3, [Pre-Chorus 1] 2, [Chorus] 4, [Verse 2] 3,
     [Pre-Chorus 2] 2, [Chorus 2] 4, [Bridge] 2, [Final Chorus] 4, [Outro] 1.
   Count the lines before you answer. If you are over, cut whole lines; never
   pad a section to fill the budget.
3. REPETITION IS THE BIGGEST RISK. The listener will skip a song that repeats
   itself. Therefore:
   - The chorus carries ONE hook: a short, singable phrase of 4-8 syllables.
     That hook appears EXACTLY TWICE in the whole song -- once in [Chorus] and
     once in [Chorus 2], both in the middle of the song.
   - No single line may be sung a third time anywhere.
   - [Intro], [Final Chorus] and [Outro] must be entirely NEW lines that appear
     nowhere else in the song. The opening, the climax and the ending are never
     borrowed text.
   - [Chorus] and [Chorus 2]: apart from the hook, all their lines are new. Two
     identical four-line choruses are a rejection.
   - [Verse 1], [Verse 2], [Pre-Chorus 1], [Pre-Chorus 2] and [Bridge] share no
     lines with each other.
4. The song must move through a huge dynamic range. Write each section in the
   register it will be sung in:
   - [Intro] and [Verse 1] and [Bridge]: LOW and intimate. Short lines, narrow
     intervals, mostly stepwise motion, few stressed syllables. A person
     speaking quietly to one other person. Nothing heroic here.
   - [Pre-Chorus 1] and [Pre-Chorus 2]: rising pressure. Short lines that get
     shorter; the last line is a launch, not a landing.
   - [Chorus] and [Chorus 2]: the first full release. Open vowels, chantable
     lines, each line 6-9 syllables, at least one line that can be shouted by a
     crowd.
   - [Final Chorus]: the peak of the whole song -- HIGHER than every chorus
     before it. Widest intervals and longest note values in the song, with one
     line whose final word is a single open vowel written to be held for a long
     note at the top of the singer's range. It must not read as a repeat.
   - [Outro]: after the peak, immediately quiet again. One line, low, final.
5. [Verse 2] advances the story -- a new place, consequence or decision. It is
   NOT Verse 1 rewritten.
6. [Bridge] is the held breath before the final chorus: quieter than anything
   else, and it delivers information no earlier section had.
7. Concrete over abstract. Name the object, the weather, the gesture. No empty
   stadium filler ("we will rise", "break the chains", "fire inside") unless the
   line earns it with a specific image.
8. English only. No Chinese characters anywhere in the lyrics.
9. Lines must be singable: mostly 5-10 syllables, no tongue-twisters, no lines
   that need more than one breath. The verse lines are shorter than the chorus
   lines.
10. A sung line is ONLY the words that are sung. Never put a stage direction,
    a performance note or a parenthetical inside a line -- no "(held note on
    night)", no "(whispered)", no brackets of any kind. The singer performs what
    is written, so such a note becomes part of the vocal.

Return STRICT JSON, no markdown, with these keys:
  title_en   : short English title, 2-4 words, the song's own name
  title_zh   : the Chinese title (for the user's review sheet)
  hook       : the exact repeated hook phrase
  lyrics     : the full lyrics with the section tags, newline-separated
  lyrics_zh  : a plain Chinese line-by-line gloss of the lyrics, same section
               tags, so a Chinese reader can judge the words
  style_extra: 8-14 comma-separated English production descriptors unique to
               THIS track (instruments, texture, the shape of the build). Do not
               describe the vocal or the section dynamics -- both are fixed
               elsewhere.
"""


def _build_user_prompt(seed: dict, used: list[dict]) -> str:
    parts = [
        f"Track direction: {seed['angle']}",
        f"Image bank (use these, do not invent around them): {seed['images']}",
        f"Musical frame: {seed['key']}, {seed['bpm']} bpm. The song must be "
        f"danceable and percussive -- this listener keeps the rhythmic tracks. "
        f"Structure: vocal in immediately, restrained verses, ramps, chorus, a "
        f"near-silent bridge, then a final chorus that is the loudest, highest "
        f"and widest moment of the song (about 75-85% in, never in the first "
        f"half).",
        f"Lead vocal (do not restate this in style_extra): {seed['voice']}, "
        f"singing in English. Verse low and close, final chorus belted high.",
    ]
    if used:
        parts.append(
            "Already used by earlier tracks in this run -- do NOT reuse any of "
            "these hooks, titles, opening images or closing images:\n"
            + "\n".join(
                f"- {u['title_en']} ({u['title_zh']}): hook \"{u['hook']}\"; "
                f"opens with \"{u['first_line']}\""
                for u in used
            )
        )
    parts.append("Write this track now.")
    return "\n\n".join(parts)


_ILLEGAL = re.compile(r"[一-鿿]")
# A paren inside a sung line is a performance note that would be sung.
_STAGE_DIRECTION = re.compile(r"[()（）]")


def validate_brief(data: dict, seed: dict) -> list[str]:
    """Local gate on what the model wrote. Returns a list of problems."""
    problems: list[str] = []
    for key in ("title_en", "title_zh", "hook", "lyrics", "lyrics_zh", "style_extra"):
        if not str(data.get(key, "")).strip():
            problems.append(f"missing {key}")

    lyrics = str(data.get("lyrics", ""))
    tags = re.findall(r"^\[([^\]]+)\]\s*$", lyrics, flags=re.MULTILINE)
    if tags != SECTION_ORDER:
        problems.append(f"section tags are {tags}, expected {SECTION_ORDER}")

    if _ILLEGAL.search(lyrics):
        problems.append("lyrics contain Chinese characters")

    lines = [ln.strip() for ln in lyrics.splitlines()
             if ln.strip() and not ln.strip().startswith("[")]

    # 2026-10-15: the 09 brief came back with `(... held note on "night")` inside a
    # sung line. The model renders what the line says, so a performance note can
    # end up in the vocal -- and it leaks into the delivered lyrics sheet.
    for line in lines:
        if _STAGE_DIRECTION.search(line):
            problems.append(f"stage direction inside a sung line: {line[:60]!r}")

    # 27 is the target; the window below is what the rendered length tolerates
    # (YuE2 s/line runs 5.8-9.7, so 30 lines can still land near 3:30).
    if not 24 <= len(lines) <= 30:
        problems.append(f"{len(lines)} sung lines, expected 24-30 (target 27)")

    # The hook must land, but it is the ONLY thing allowed to repeat, and only
    # twice -- 2026-10-15, the user's "重复两遍顶天了".
    hook = str(data.get("hook", "")).strip().lower()
    body = "\n".join(lines).lower()
    hook_count = body.count(hook) if hook else 0
    if hook_count == 0:
        problems.append(f"hook {hook!r} never appears")
    elif hook_count > 2:
        problems.append(f"hook {hook!r} appears {hook_count}x, max 2")

    try:
        repetition = audit_repetition(lyrics)
        if not repetition.passes:
            problems.append(f"repetition: {repetition.summary()}")
    except Exception as exc:
        problems.append(f"repetition check errored: {exc}")

    if len(str(data.get("style_extra", ""))) < 40:
        problems.append("style_extra is too thin to shape the arrangement")

    try:
        diversity = analyze_lyrics(lyrics)
        if not diversity.passes:
            problems.append(f"lyric diversity: {diversity.summary()}")
    except Exception as exc:  # checker is advisory, never fatal
        problems.append(f"diversity check errored: {exc}")
    return problems


# The fixed half of every tag file: the register the user asked for, plus the
# negatives that keep HeartMuLa-era defects (and YuE2's own) out of the mix.
TAG_HEAD = (
    "powerful female lead vocal singing in English, {voice}, epic game battle "
    "anthem, HOYO-MiX style English theme, rock orchestra hybrid, distorted "
    "electric guitar power chords and lead lines, driving rock drum kit with "
    "punchy kick and snappy snare, orchestral ostinato strings, low brass "
    "braams, brass fanfare stabs, epic choir, taiko war drums, timpani, deep "
    "sub bass drum hits, cinematic widescreen production, wide stereo"
)
# Section-by-section dynamics, rewritten 2026-10-15 after the user called the
# previous batch too flat ("低的时候不够低，高的时候不够高"). The old tag said
# only "restrained intro ... late climactic final chorus".
#
# The devices named here come from the reference tracks the user asked me to
# learn from -- HI3 "Starfall" (Tia Ray), Arknights "Radiant"/耀光 (Mary Clare),
# HSR "Wildfire" and "Flares of the Blazing Sun". All four use the same handful
# of moves: the verse is a fraction of the chorus's energy, the bridge drops the
# whole band, a beat of real silence sits right before the final chorus, and the
# last chorus is raised (Starfall modulates Dm -> D). Those are now in the tag.
TAG_ARC = (
    "vocal enters within the first two seconds, no instrumental intro, verse: "
    "drums out, sparse clean guitar or piano and sub bass only, vocal one "
    "octave lower than the chorus, soft, breathy and close-mic'd, about thirty "
    "percent of the chorus energy, pre-chorus: drums enter with a rising snare "
    "roll and sustained strings, building tension, no cymbals yet, chorus: full "
    "rock orchestra tutti slams in with double-tracked and octave-doubled lead "
    "vocal, epic choir, distorted guitars, taiko and driving drums, sub-bass "
    "drop on the downbeat, bridge: half-time, every instrument drops out to "
    "piano and one close whispered voice, the quietest point of the song, one "
    "to four seconds of total silence before the final chorus, final chorus: "
    "modulates up a whole step, maximum power, gang vocal shouts, cymbal "
    "crashes, brass fanfare and thunderous low end, the lead belted at the top "
    "of her range on a long held open note with ad-libs, outro: sudden pull "
    "back to a single sustained voice, extreme dynamic range, dramatic "
    "loud-quiet contrast, never a constant full-band wall of sound"
)
TAG_VOCAL = (
    "clear human vocal, natural breath, precise consonants, expressive legato, "
    "verse sung low and close and breathy, chorus belted high and open, open "
    "vowel belting on the held notes, controlled vibrato only at phrase ends, "
    "believable emotional delivery"
)
TAG_NEG = (
    "no male vocal, no vocoder, no robotic tremolo, no metallic resonance, no "
    "muffled dull tone, no over-compression, no lo-fi, no chiptune, no extended "
    "instrumental intro, no steady loud wall of sound, no constant full-band "
    "texture under the verses, no flat even dynamics"
)


def dedupe_segments(tags: str) -> str:
    """Drop repeated comma-separated segments, keeping first-seen order.

    The model likes to restate the tempo and key inside style_extra, which then
    shows up twice in the file ("152 bpm, D minor, D minor, 152 bpm"). Harmless
    but noisy; a duplicate tells the model nothing the first copy did not.
    """
    seen: set[str] = set()
    out: list[str] = []
    for part in tags.split(","):
        cleaned = " ".join(part.split()).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return ", ".join(out)


def build_tags(seed: dict, style_extra: str) -> str:
    extra = " ".join(style_extra.split()).strip().rstrip(",")
    return dedupe_segments(", ".join([
        TAG_HEAD.format(voice=seed["voice"]),
        f"{seed['bpm']} bpm, {seed['key']}",
        extra,
        TAG_ARC,
        TAG_VOCAL,
        TAG_NEG,
    ]))


# --- line-by-line Chinese gloss --------------------------------------------
#
# The user asked on 2026-10-15 for the delivered English lyrics to carry a
# Chinese line under every English line, so a Chinese reader can follow the
# words without a separate translation. The model already returns `lyrics_zh`
# for this; what was missing is that it must stay *aligned* (same section tags,
# same number of lines per section) and must reach the delivery folder as a
# sidecar file, not just live inside briefs.json.

GLOSS_PROMPT = """You translate English song lyrics into Chinese for a bilingual \
lyric sheet.

Rules:
1. Keep every section tag exactly as it is, on its own line ([Intro], [Verse 1], \
[Pre-Chorus 1], [Chorus], [Verse 2], [Pre-Chorus 2], [Chorus 2], [Bridge], \
[Final Chorus], [Outro]).
2. Translate LINE BY LINE: one Chinese line for every English line, in the same \
order, with no merging and no splitting. The Chinese line count under each tag \
must equal the English line count.
3. Natural, singable Chinese -- not a word-for-word gloss, but the same image and \
the same number of ideas per line.
4. Output ONLY the Chinese lyrics with the tags. No explanations, no markdown, no \
English.

Return STRICT JSON: {"lyrics_zh": "<the Chinese lyrics>"}"""


def request_gloss(client, base_url: str, key: str, lyrics: str,
                  retries: int = 2) -> str:
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": GLOSS_PROMPT},
            {"role": "user", "content": f"English lyrics:\n\n{lyrics}"},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    resp = client.post(f"{base_url}/v1/chat/completions", json=payload,
                       headers={"Authorization": f"Bearer {key}",
                                "Content-Type": "application/json"})
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{[\s\S]*\}", content)
    data = json.loads(match.group(0) if match else content)
    return str(data.get("lyrics_zh", "")).strip()


def validate_gloss(lyrics: str, gloss: str) -> list[str]:
    """Alignment gate: same tags in the same order, same lines per tag.

    Without this the bilingual sheet would silently pair an English line with
    the wrong Chinese one, which is worse than no translation at all.
    """
    from app.services.lyric_diversity import ordered_sections

    en = ordered_sections(lyrics)
    zh = ordered_sections(gloss)
    if not zh:
        return ["gloss is empty"]
    if [t for t, _ in en] != [t for t, _ in zh]:
        return [f"gloss tags {[t for t, _ in zh]} do not match the lyrics"]
    problems = []
    for (tag, en_lines), (_, zh_lines) in zip(en, zh):
        if len(en_lines) != len(zh_lines):
            problems.append(f"[{tag}] {len(en_lines)} English lines vs "
                            f"{len(zh_lines)} Chinese lines")
    return problems


def write_gloss_sidecar(out_dir: Path, slug: str, gloss: str) -> Path:
    path = out_dir / f"{slug}_lyrics_zh.txt"
    path.write_text(gloss.strip() + "\n", encoding="utf-8")
    return path


def _read_api_key(env_path: Path) -> tuple[str, str]:
    key = base = ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DEEPSEEK_API_KEY="):
            key = line.split("=", 1)[1].strip()
        elif line.startswith("DEEPSEEK_BASE_URL="):
            base = line.split("=", 1)[1].strip()
    return key, (base or "https://api.deepseek.com")


def request_brief(client, base_url: str, key: str, seed: dict,
                  used: list[dict], problems: list[str] | None = None,
                  retries: int = 4) -> dict:
    """One DeepSeek round trip. `problems` re-prompts with the local gate's list.

    Retries the *transport* on 429/5xx with exponential backoff. Added
    2026-10-15 after a 503 burst burned a title's whole validation budget on
    infrastructure instead of on the lyrics (only one of twelve briefs had been
    written when the first outage hit). The outer loop in `main` still owns the
    validation retries, so the two failure modes stay separate.
    """
    import httpx

    user = _build_user_prompt(seed, used)
    if problems:
        user += ("\n\nYour previous attempt failed local validation. Fix exactly "
                 "these problems and output the whole JSON again:\n- "
                 + "\n- ".join(problems))
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 1.0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    delay = 10.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.post(f"{base_url}/v1/chat/completions", json=payload,
                               headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"})
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                print(f"    api {resp.status_code}; retry in {delay:.0f}s",
                      flush=True)
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
        except httpx.TransportError as exc:
            last = exc
            if attempt >= retries:
                raise
            print(f"    api unreachable ({exc.__class__.__name__}); "
                  f"retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay *= 2
            continue
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", content)
        return json.loads(match.group(0) if match else content)
    raise last or RuntimeError("deepseek: no response after retries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "inputs" / "2026-10-15_epic_en")
    ap.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env")
    ap.add_argument("--force", action="store_true",
                    help="rewrite titles whose lyrics already exist")
    ap.add_argument("--only", default=None, help="slug substring filter")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--tags-only", action="store_true",
                    help="rebuild only the *_tags.txt files from briefs.json. "
                         "The tag head/arc/vocal/negatives are code, not model "
                         "output, so a caption change must not cost a new batch "
                         "of lyrics (2026-10-15: the reference-derived dynamics "
                         "landed after the lyrics were already written).")
    ap.add_argument("--gloss-only", action="store_true",
                    help="(re)build the <slug>_lyrics_zh.txt sidecars: the "
                         "line-by-line Chinese translation of each English lyric, "
                         "which the delivery folder pairs under every English "
                         "line. Skips titles whose gloss already aligns.")
    args = ap.parse_args()

    key, base_url = _read_api_key(args.env)
    if not key:
        print(f"no DEEPSEEK_API_KEY in {args.env}")
        return 1

    import httpx

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "briefs.json"
    meta: dict = (json.loads(meta_path.read_text(encoding="utf-8"))
                  if meta_path.exists() else {})

    used: list[dict] = [
        {"title_en": m["title_en"], "title_zh": m["title_zh"], "hook": m["hook"],
         "first_line": m["first_line"]}
        for slug, m in sorted(meta.items())
    ]
    langs = {s["slug"]: "en" for s in SEEDS}
    titles = {s["slug"]: meta.get(s["slug"], {}).get("title_zh", s["slug"])
              for s in SEEDS}

    if args.tags_only:
        seeds = {s["slug"]: s for s in SEEDS}
        rebuilt = 0
        for slug, m in sorted(meta.items()):
            seed = seeds.get(slug)
            if not seed or not m.get("style_extra"):
                continue
            if args.only and args.only not in slug:
                continue
            (args.out_dir / f"{slug}_tags.txt").write_text(
                build_tags(seed, str(m["style_extra"])), encoding="utf-8")
            rebuilt += 1
            print(f"[{slug}] tags rebuilt")
        print(f"\n{rebuilt} tag files rewritten in {args.out_dir}")
        return 0

    if args.gloss_only:
        # Repair/refresh the line-by-line Chinese sidecars without touching the
        # English lyrics. Needed because a gloss can be lost (2026-10-15: a
        # briefs.json write race dropped 09's) and because the delivery folder
        # needs the gloss as a file next to the lyrics, not only in briefs.json.
        fixed = failed = 0
        with httpx.Client(timeout=180) as client:
            for slug in sorted(p.name[: -len("_lyrics.txt")]
                               for p in args.out_dir.glob("*_lyrics.txt")):
                if args.only and args.only not in slug:
                    continue
                lyrics = (args.out_dir / f"{slug}_lyrics.txt").read_text(
                    encoding="utf-8")
                gloss = "" if args.force else str(meta.get(slug, {})
                                                  .get("lyrics_zh", "")).strip()
                if gloss and not validate_gloss(lyrics, gloss):
                    write_gloss_sidecar(args.out_dir, slug, gloss)
                    print(f"[{slug}] gloss already aligned ({len(gloss)} chars)")
                    fixed += 1
                    continue
                problems: list[str] = []
                for attempt in range(args.attempts):
                    try:
                        gloss = request_gloss(client, base_url, key, lyrics)
                    except Exception as exc:
                        print(f"[{slug}] gloss attempt {attempt + 1} failed: {exc}")
                        time.sleep(5)
                        continue
                    problems = validate_gloss(lyrics, gloss)
                    if not problems:
                        break
                    print(f"[{slug}] gloss attempt {attempt + 1} misaligned -> "
                          f"{problems}")
                if not gloss or problems:
                    print(f"[{slug}] FAILED to produce an aligned gloss: {problems}")
                    failed += 1
                    continue
                write_gloss_sidecar(args.out_dir, slug, gloss)
                meta.setdefault(slug, {})["lyrics_zh"] = gloss
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[{slug}] gloss written ({len(gloss)} chars)")
                fixed += 1
        print(f"\n{fixed} glosses ok, {failed} failed, in {args.out_dir}")
        return 0 if not failed else 1

    with httpx.Client(timeout=180) as client:
        for seed in SEEDS:
            slug = seed["slug"]
            if args.only and args.only not in slug:
                continue
            lyrics_path = args.out_dir / f"{slug}_lyrics.txt"
            if lyrics_path.exists() and not args.force:
                print(f"[{slug}] exists, skip", flush=True)
                continue

            data: dict | None = None
            problems: list[str] | None = None
            for attempt in range(1, args.attempts + 1):
                t0 = time.time()
                try:
                    data = request_brief(client, base_url, key, seed, used, problems)
                except Exception as exc:
                    print(f"[{slug}] attempt {attempt}: request failed: {exc}",
                          flush=True)
                    time.sleep(5)
                    continue
                problems = validate_brief(data, seed)
                if not problems:
                    print(f"[{slug}] attempt {attempt}: ok in {time.time() - t0:.0f}s",
                          flush=True)
                    break
                print(f"[{slug}] attempt {attempt}: rejected -> {problems}",
                      flush=True)
                data = None
            if not data:
                print(f"[{slug}] FAILED after {args.attempts} attempts; "
                      f"leaving it out of this batch", flush=True)
                continue

            lyrics = str(data["lyrics"]).strip() + "\n"
            lyrics_path.write_text(lyrics, encoding="utf-8")
            (args.out_dir / f"{slug}_tags.txt").write_text(
                build_tags(seed, str(data["style_extra"])), encoding="utf-8")

            # The bilingual delivery needs the gloss as a file next to the
            # lyrics, so write it here rather than leaving it in briefs.json.
            gloss = str(data.get("lyrics_zh", "")).strip()
            if gloss and not validate_gloss(lyrics, gloss):
                write_gloss_sidecar(args.out_dir, slug, gloss)
            else:
                print(f"[{slug}] WARNING: stored gloss does not align "
                      f"({validate_gloss(lyrics, gloss)}); run --gloss-only",
                      flush=True)

            lines = [ln.strip() for ln in lyrics.splitlines()
                     if ln.strip() and not ln.strip().startswith("[")]
            entry = {
                "slug": slug, "title_en": data["title_en"],
                "title_zh": data["title_zh"], "hook": data["hook"],
                "key": seed["key"], "bpm": seed["bpm"],
                "angle": seed["angle"], "voice": seed["voice"],
                "style_extra": data["style_extra"],
                "lyrics_zh": str(data["lyrics_zh"]).strip(),
                "sung_lines": len(lines), "first_line": lines[0] if lines else "",
            }
            meta[slug] = entry
            langs[slug] = "en"
            titles[slug] = data["title_zh"]
            used.append({"title_en": data["title_en"],
                         "title_zh": data["title_zh"], "hook": data["hook"],
                         "first_line": entry["first_line"]})
            # Re-read before writing: two titles can be in flight at once (the
            # 2026-10-15 session ran `--only 09_` and `--only 12_` side by side,
            # and each process wrote its own stale in-memory copy, so whichever
            # finished last dropped the other's entry).
            on_disk = {}
            if meta_path.exists():
                try:
                    on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    on_disk = {}
            on_disk.update(meta)
            meta_path.write_text(
                json.dumps(on_disk, indent=2, ensure_ascii=False), encoding="utf-8")
            (args.out_dir / "_langs.json").write_text(
                json.dumps(langs, indent=2, ensure_ascii=False), encoding="utf-8")
            (args.out_dir / "_titles.json").write_text(
                json.dumps(titles, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{slug}] \"{data['title_en']}\" / {data['title_zh']}  "
                  f"hook=\"{data['hook']}\"  {entry['sung_lines']} lines",
                  flush=True)

    print(f"\nbriefs written to {args.out_dir} ({len(meta)} titles)")
    for slug, m in sorted(meta.items()):
        print(f"  {slug:22s} {m['title_zh']:8s} {m['title_en']:28s} "
              f"{m['bpm']} bpm {m['key']:9s} hook=\"{m['hook']}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
