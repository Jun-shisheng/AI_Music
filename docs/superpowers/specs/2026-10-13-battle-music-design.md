# Stage 3 design: game battle / BGM quality (2026-10-13)

## Goal

Move the pipeline from "accepted vocal pop songs" to **Chinese gacha-game /
battle-anime soundtrack level** (原神·崩铁·明日方舟 / Final Fantasy-class battle
and BGM cues), on local hardware (RTX 3060 12 GB, 32 GB RAM), 48 kHz, with the
project's source-clean principle intact (no post-processing repair).

User decision on the first listening round: **vocal battle tracks** ("带人声的
战斗曲"), with the instrumental version of the same brief kept as a control.

## Why the backend has to change (the central constraint)

HeartMuLa is not tunable into this target:

- HeartCodec runs at a **hard 12.5 Hz frame rate** (80 ms/frame), RVQ K=8,
  V=8192, 48 kHz stereo, ~3840x compression, autoregressive tokens.
- `HeartMuLa-oss-3B-happy-new-year` (13 Feb 2026) changed the **LM only**;
  `HeartCodec-oss-20260123` only "optimizes audio decoding quality". The 7B
  model is still unreleased (as of 2026-10-13).
- Consequence: fast transients are structurally smeared. This is the origin of
  the residual crackle/breaks this project spent months chasing; no
  post-processing removed it because the information is not in the tokens.

Everything else in this document follows from that: pick backends whose audio
rate is finer, and treat HeartMuLa as the quality **baseline** to beat.

## Candidate landscape (verified against the HF mirror on this machine)

Sourcing caveat: WebFetch was blocked for every external domain in this
environment, so the two research agents worked from search-result summaries.
Every repo below was then re-checked by me directly against `hf-mirror.com`
(existence, gated flag, file sizes). Quality/benchmark claims remain
second-hand and are marked as such.

| Model | Date | License | Size / VRAM | Audio rate | Verdict |
|---|---|---|---|---|---|
| **YuE2-3B** (`m-a-p/YuE2-3B`) | Oct 2026 | CC BY-NC 4.0 + individual-creator carve-out | 7.26 GB single safetensors; fits 12 GB bf16 | 48 kHz, **25 fps**, symbolic ABC planning first | **#1 for this goal** |
| **ACE-Step 1.5** (2B turbo / XL 4B) | Feb / Apr 2026 | **MIT** | 2B stack already local (~11 GB peak); XL 4B needs offload+quant at 12 GB | 48 kHz, 25 Hz VAE latents | **#2**, and the fast iteration loop |
| SongGeneration 2 / LeVo 2 | Mar 2026 | ambiguous, treat as non-commercial | v2-large 22 GB (v2-medium 12 GB not released) | 48 kHz, ~25 fps | best native stem/split output; heavier |
| MiniMax-Music3 | Aug 2026 | custom | 22-24 GB | 32 kHz | too heavy for 12 GB |
| Stable Audio 3 Medium | May 2026 | Stability Community | 1.4B, ~6.5 GB | 44.1 kHz, ~10.8 Hz latents | instrumental/cinematic companion only |
| DiffRhythm 2 | Oct 2025 | Apache-2.0 | — | 48 kHz, 5 Hz | quality too low to be an upgrade |
| MusicGen / MAGNeT | 2023-24 | — | — | 32 kHz | frozen, superseded |
| Riffusion / Lyria | dead / closed | — | — | — | not usable |

On-target LoRAs, both verified present and ungated:

- `monsterovich/yue2-steps-from-hell` (0.23 GB) — Two Steps From Hell-style epic
  orchestral trailer library (heroic strings, taiko, brass, choir). Recommended
  `ar_scale≈0.5 / nar_scale≈0.5`.
- `Mothersuperior/YuE2-instrumental-cot-full-loras` (0.63 GB) — instrumental
  workflow LoRA.

Benchmark caveat: YuE2's #1 SongBench position is best-of-8 and single-source;
the honest reading is "top tier, unverified by us" until we audition it.

## What the genre actually requires (research summary)

**Arrangement.** A layered stack: live-scale orchestra (strings as the ostinato
engine, brass as the harmonic lift) + real choir (often texted; Chu Ci style,
Latin, invented languages) + synth bass/arps + programmed drums + taiko/world
percussion, plus one region-specific folk instrument on top (erhu, pipa,
shakuhachi, ney, duduk). Reference structure: intro sting -> brass lift ->
ostinato loop -> climactic cadence/key change; ~100-150 BPM, minor/modal, often
avoiding functional V cadences (e.g. Liyue battle material keeps the VI chord
minor so it never resolves "victoriously").

**Vocals vs instrumental.** In-game battle BGM is instrumental; the vocal
version is a separate artefact (phase-2 boss songs such as HSR "Wildfire",
character demo themes, anniversaries). When vocals are present they are either
a powerful female pop/pop-classical lead (Sawano's vocalists; Arknights
commissions Aimer, Machico) or layered/ethnic choirs.

**Mix.** Game-embedded audio targets ~-23 LUFS with -1 dBTP; OST album releases
sit near streaming loudness (-14 LUFS). Trailer-library music is far hotter.
Broadcast orchestral DR is roughly 10-14. These are genre standards, not
measurements of any specific HOYO-MiX/MSR master (none found).

**Why AI orchestral music sounds AI** (the tells to design against):

1. Strings are the single biggest giveaway — the model synthesises the *outcome*
   of a performance, not the performance: no bow changes, slurs or portamento,
   and implied bowing that is physically unplayable.
2. Top end "behind frosted glass": everything above ~10 kHz is too even, and
   consonants/transients lose their edges.
3. No true silence / no 留白; every band is uniformly full.
4. Transients too uniform — identical peak shape on every hit, no dynamics
   pull-back between sections.
5. Fake reverb space (inconsistent tails), "width without placement".
6. Long-range coherence decays past ~2.5-3.5 min (chunked generation).

**Prompt vocabulary that maps to this genre** (compiled, paste-ready):
`epic orchestral hybrid, cinematic trailer music, battle theme, boss theme,
ostinato strings, brass fanfare, low brass braams, taiko, war drums, timpani,
epic choir, sub bass, synth pulse, riser, impact hit, minor key, D minor,
A aeolian, dorian, phrygian, 100-150 BPM, heroic, defiant, climactic, ominous`.

## Local environment facts (learned the hard way today)

- `huggingface.co` is **unreachable** here; `hf-mirror.com` works. `hf_hub`
  0.36 in the ACE venv fails its HEAD probe against the mirror (the Xet
  transport path is unusable) and reports a misleading
  `LocalEntryNotFoundError`, even though plain `httpx`/`curl` succeed.
  Workaround: `tools/generation/download_models_yue2.py` downloads with
  `curl -L --retry 5 -C -`, which also resumes.
- ACE-Step 1.5 is already fully vendored at `vendor/ACE-Step-1.5`
  (`acestep-v15-turbo` 4.5 GB, 5 Hz LM 0.6B + 1.7B, vae,
  Qwen3-Embedding-0.6B). `vllm` cannot be installed on Windows, so the LM
  backend must be `pt`; the repo `.env` still says `vllm`, so pass it
  explicitly.
- ACE-Step writes 48 kHz `wav` as 32-bit float stereo. Batch of 2 x 60 s
  takes: **120-150 s** end to end on the 3060 with `offload` enabled.
- HeartMuLa 60 s single pass: **~4.5 min per take**, ~10.7 GB peak, cfg 2.0.
- Process hygiene that bit twice today: a long caption passed through
  bash -> PowerShell -> `Start-Process` loses characters, and printing ACE's
  emoji init message through a redirected GBK console kills the run. Both are
  fixed in the tooling (`--job` JSON; explicit UTF-8 stream reconfigure).

## Tooling added

- `tools/generation/acestep_generate.py` — source-clean ACE-Step backend
  (caption + optional bpm/key/time-signature + lyrics; no post-processing).
  Takes a `--job` JSON so long captions survive the shell.
- `tools/generation/download_models_yue2.py` — resumable curl-based downloader
  for the YuE2 stack.
- Briefs live in `inputs/battle_test/` (`01_*` instrumental, `02_*` vocal),
  one pair of files per backend (HeartMuLa tags+lyrics; ACE job JSON).

## First A/B results (60 s, D minor, 140 BPM, same brief)

Objective diagnostics only — the human still picks the take.

| take | clicks | both-ch | dips | harsh | centroid Hz | 8-16k band | HNR |
|---|---|---|---|---|---|---|---|
| ACE instrumental c1 | 3012 | 281 | 2 | 0 | 1863 | -7.4 dB | 5.33 |
| ACE instrumental c2 | 13726 | 2752 | 1 | 0 | 2157 | -4.9 dB | 5.36 |
| HeartMuLa instrumental c1 | 7198 | 1390 | **9** | 0 | 2724 | **+0.5 dB** | 2.87 |
| HeartMuLa instrumental c2 | — | — | — | 0 | 2929 | **+0.4 dB** | 3.30 |
| ACE vocal c1 | 4192 | 409 | 2 | 0 | 2717 | -1.9 dB | 3.86 |

Readings worth keeping:

- The `clicks` metric **saturates on taiko-heavy material** (it was calibrated
  on vocal pop); it cannot gate this genre. `dips` is the column that tracks the
  known defect, and HeartMuLa shows 9 in 60 s against ACE's 1-2.
- HeartMuLa carries ~5-8 dB more energy in the 8-16 kHz band than ACE
  instrumental and a much higher centroid, with HNR 2.9-3.3 vs ACE's 5.3. That
  is the same bright-band signature the old repair chain fought for months, and
  it is consistent with a coarse codec adding HF hash. Confounded by
  arrangement differences, so treat as supporting evidence, not proof.
- Zero `harsh` frames on every take, including HeartMuLa — that proxy was
  designed for separated vocal stems and says little about a full orchestral
  mix.

## Plan

- **Phase A (done today):** re-establish the research, verify all download
  targets against the live mirror, get ACE-Step 1.5 running as a local backend,
  and produce the first instrumental + vocal battle A/B on both existing
  backends.
- **Phase B:** finish the YuE2-3B download, wire up its inference (the HF repo
  ships `modeling_yue2.py` plus a `yue2_infer` wheel, so the full YuE repo is
  not required), and re-run the same brief through it — including the
  Two Steps From Hell and instrumental LoRAs.
- **Phase C:** three-way listening selection on the vocal brief; only then
  decide which backend becomes the battle-music default and whether to invest
  in ACE-Step XL (4B) or its LoRA training path.
- **Phase D (if the model side stalls):** the genre's own answer to mix quality
  is stem-wise production — generate, separate stems, re-mix and shape
  transients. ACE-Step has built-in separation, so this stays local.

## Open questions

- Does the vocal battle path need the lead-vocal + choir stack in the tags, or
  does the model handle that implicitly? First listening round decides.
- Intonation/diction check (the project has repeatedly hit "unclear diction")
  should be measured with a transcription pass; no ASR is installed in any
  venv yet.
- YuE2 weights are CC BY-NC: fine for this project's personal/non-commercial
  use, but it blocks any future commercial release.
