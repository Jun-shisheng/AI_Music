# AI Music Studio

**A local-first pipeline for producing complete songs on your own GPU** — an LLM
writes the lyrics, a local music model renders the audio, every take is measured
before it is believed, and what survives is packaged into a dated delivery folder.

No cloud upload of your music. No account. The only external call is the optional
LLM step that writes lyrics.

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.13-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%2012GB%2B-orange" alt="NVIDIA GPU 12GB+">
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey" alt="Windows">
</p>

---

## The pipeline

```
   idea / brief
        │
        ▼
  ┌───────────────┐   lyrics + section tags + style caption
  │ LLM write     │   section-quota prompts, repetition audit,
  │ (optional)    │   line-by-line translation sidecar
  └───────┬───────┘
          ▼
  ┌───────────────┐   symbolic plan ──► audio
  │ local model   │   (one process per title, resume-safe,
  │ render        │    multi-candidate with different seeds)
  └───────┬───────┘
          ▼
  ┌───────────────┐   arc · diction · dynamics · clicks · dropouts
  │ measure       │   → candidate ranking + hard gates
  └───────┬───────┘
          ▼
  ┌───────────────┐   WAV + same-stem lyrics TXT + manifest
  │ package       │   + a listening sheet a human can actually use
  └───────────────┘
          ▼
      human ears decide
```

The interesting part of this project is not the generation — it is the
**measurement**. Music models sound plausible in a spectrogram and wrong in your
ears, so every claim here is a number you can re-derive: a dynamic arc measured in
dB, a sung-word rate measured with word-level timestamps, a pitch curve, a
candidate ranking that a person can audit. Prompt text cannot control a
song's dynamic range — that was tried and measured — so the pipeline measures
instead of hoping, and the human stays the final judge.

## Repository layout

| Path | What it is |
|---|---|
| `tools/generation/` | Measurement, QA and packaging utilities for a render batch |
| `tools/vocal_repair/` | An earlier experiment in repairing vocal artifacts (deprecated; kept for the record) |
| `services/api/` | FastAPI backend of the earlier "studio" prototype (arrange → queue → render) |
| `apps/web/` | Next.js front end of the same prototype |
| `scripts/` | Model weight download helpers |
| `docs/` | Dev log, workflow notes, and dated design records |

**Audio, model weights and song data are deliberately not in this repository.**
`.gitignore` refuses `outputs/`, `inputs/`, `logs/`, `human_refs/`, `models/`,
`vendor/`, `data/`, `.cache/` and every audio extension. Bring your own.

## The measured toolkit

Everything under `tools/generation/` is a CLI with no hidden state, so results can
be compared between sessions and between takes.

| Tool | Question it answers |
|---|---|
| `arc_qa.py` | Is there a real dynamic arc? Opening level, total lift in dB, where the climax sits on the timeline, how low the body sits under its own peak, rhythmic onset density. Prints an ASCII envelope so the shape is visible without listening. |
| `diction_qa.py` | Did the model actually sing the words? Transcribes a take with faster-whisper and scores it against the intended lyric: character-level ratio, how many lines came back, how much text is missing, and when the first sung word lands. |
| `audio_qa.py` | Is the file broken? Sample-level clicks, dropout gaps, and a zero-crossing-based harsh-frame count — a diagnostic table, explicitly **not** an acceptance gate. |
| `loudness.py` | Linear loudness normalisation that does not touch the spectrum. |
| `export_delivery.py` | Builds the dated delivery folder: one WAV per take, a same-stem lyrics file (section tags once, each translated line under its source line), and a `manifest.json` carrying every take's own numbers. |
| `battle_review.py` | Renders the listening sheet: the metrics with an explanation of how to read them, plus a short checklist. |
| `services/api/app/services/lyric_diversity.py` | Rejects mechanically repeated lyrics — a hook is allowed to return, a wall of identical lines is not. |

Typical use:

```bash
# how does this take move?
python -I tools/generation/arc_qa.py outputs/mybatch/selected/01_song.wav

# did it sing the words?
python -I tools/generation/diction_qa.py --lyrics inputs/01_song_lyrics.txt \
    --language en --out logs/diction.md outputs/mybatch/selected/01_song.wav

# package a finished batch
python -I tools/generation/export_delivery.py \
    --batch-dir outputs/mybatch --engine Local --name mybatch
```

`diction_qa.py` expects a faster-whisper model on disk (`--model`, default
`PR_AIMusic_Extension/models/faster-whisper-small`); download one with
`faster-whisper`'s own tooling and point `--model` at it. Everything else needs
only `numpy` and `soundfile`.

## Hardware reality (12 GB card, measured)

Notes worth having before you spend a night on this:

- **A 12 GB card is the practical floor** for a full-length song with a
  high-fidelity codec. Peak memory is driven by the decode stage, not by song
  length, so a 3-minute song and a 6-minute song can peak the same.
- **The KV cache, not the song, decides whether you can render.** If the model
  preallocates for its maximum sequence length, right-sizing it to prompt+frames
  frees gigabytes and is usually the difference between "fits" and "OOM".
- **One process per title.** A long-lived process fragments the CUDA allocator
  across generations and eventually fails during a large single allocation.
  Paying one model load per title removes that failure mode entirely.
- **`expandable_segments` is not available on Windows.** Process hygiene is the
  workaround, not configuration.
- **Close anything else that holds VRAM** (a resident image-generation UI will
  happily sit on 10 GB and starve a render).
- **A resident model owns RAM, not just VRAM.** Free physical memory collapsed to
  under 2 GB in one measured session; a render that normally takes 6.5 minutes
  took 29 because of it. The audio was unaffected — token generation is
  determined by step count, not wall time.
- Sample-level click counts are **useless as a defect metric in percussion-heavy
  music**: drums are transients by definition. Measured across one delivery, the
  click rate varied 100× between titles and none of it was a defect.

## Acceptance philosophy

Three rules that came out of building this, each of them the hard way:

1. **Measure, then listen — and never let a proxy overrule an ear.**
   Spectral proxies cannot tell a sibilant from machine noise, and a model that
   scores perfectly can still sound synthetic. Numbers rank candidates; a person
   picks the take.
2. **Fix the source, not the symptom.** A long chain of post-processing (frame
   repair, EQ shelves, formant shaping) traded one artifact for another and hit a
   hard ceiling. Source-clean generation plus candidate selection beat it.
3. **When a metric and a listener disagree, the metric is wrong until proven
   otherwise.** Anchor the number to the listener: have them point at the exact
   seconds that sound wrong, measure those frames against the rest of the track,
   find the statistic that separates the two groups, then gate on it. That loop
   is the only way a proxy earns trust.

## Getting started with the studio prototype

`services/api` + `apps/web` are a front end/back end pair around a local model:
arrange with an LLM → queue → render → library, all on `127.0.0.1`. They are kept
as a working reference for the interactive path; the measured pipeline above is
what this project actually runs for deliveries.

**The API contract is written down in [`docs/api.md`](docs/api.md)** — every
endpoint, its payloads, the `generation_type` mapping, and the failure semantics.
Change one side of the interface by changing that document too.

```bash
cp .env.example .env         # set DEEPSEEK_API_KEY for /arrange; .env is gitignored
pip install -r services/api/requirements-dev.txt   # runtime + test deps

# 1. a model checkout + weights next to the repo (not versioned; ~10 GB)
python scripts/download_models.py
# 2. the model's own API server on 127.0.0.1:8765 (the default ACESTEP_BACKEND=api)
# 3. tests, backend, then front end
cd services/api && python -m pytest -q
cd services/api && python -m uvicorn app.main:app --port 8000
cd apps/web && npm install && npm run dev
```

After installing the API dependencies in the root `.venv` (or
`services/api/.venv`), the ACE-Step checkout and its own `.venv`, and the frontend
dependencies, `start.bat` downloads missing model weights and starts the three
services on Windows. It locates the repository from its own path; set
`AI_MUSIC_ROOT` to use another repository root. Missing prerequisites, occupied
ports, and readiness timeouts stop startup with an error. Existing processes are
never killed automatically. The launcher and `npm run dev` / `npm start` bind
the frontend to `127.0.0.1`.

### Three kinds of generation

`POST /generate` takes a `generation_type`, and the back end turns it into what
the model is actually asked for:

| `generation_type` | what it produces | how the request differs |
|---|---|---|
| `song` | a sung song (or an instrumental, via `instrumental: true`) | nothing added |
| `bgm_loop` | seamlessly looping background music | instrumental, `seamless loop / loopable / clean loop point` |
| `sfx` | a sound effect | non-musical, no vocals, 10 s unless a duration is given |

So one API covers **music and sound effects**; the mapping lives in one place
(`resolve_request()` in `services/api/app/services/acestep_runner.py`) and is
covered by tests, so the UI cannot promise a type the pipeline ignores.

## License

MIT — see `LICENSE`. Model weights carry their own licenses; check the model you
download.
