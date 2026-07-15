# AI Music Studio

**Local-first AI music creation workspace.** Structure lyrics via LLM, generate full songs on your own GPU — no cloud uploads, no subscriptions, no accounts.

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.13-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%208GB%2B-orange" alt="NVIDIA GPU 8GB+">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey" alt="Windows | Linux">
</p>

## Overview

AI Music Studio is a three-stage music creation pipeline that runs entirely on your local machine:

1. **Arrange (Studio)** — Describe your song idea in natural language. DeepSeek generates structured lyrics with section tags and style metadata.
2. **Create (Compose)** — Fine-tune lyrics and configure genre, mood, instruments, vocals, and tempo via an interactive UI.
3. **Generate (Render)** — ACE-Step 1.5 renders the final audio on your local GPU with Chinese vocal support.

The only external API call is the optional DeepSeek arrangement step. Music generation is 100% local.

## Features

- **Natural language songwriting assistance** — prompt → structured lyrics + style tags via DeepSeek
- **Interactive style configuration** — 18 genres, 16 moods, 18 instruments, 8 vocal types, 6 tempo ranges with visual chip selectors
- **Song structure toolbar** — one-click insert `[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]` and more
- **Local GPU inference** — ACE-Step 1.5 model runs on your own hardware, zero cloud dependency
- **Real-time job status** — polling widget shows generation progress with audio preview and download
- **Local music library** — browse, filter, play, and download all generated songs
- **Dark theme UI** — distraction-free workspace with Tailwind CSS
- **One-click launcher** — single `start.bat` handles model download, service startup, and browser launch

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU, 8 GB VRAM | 12+ GB VRAM |
| RAM | 16 GB | 32 GB |
| Disk | 30 GB free | 50+ GB (model weights ~11 GB) |
| OS | Windows 10+ / Linux (x86_64) | — |
| CUDA | 12.4+ | 12.8 |

**Verified on:** NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM), Windows 11, Python 3.11 (ACE-Step) / 3.13 (API).

## Quick Start

### Prerequisites

- NVIDIA GPU with ≥8 GB VRAM and CUDA 12.4+ drivers
- Python 3.11–3.13
- Git
- [DeepSeek API key](https://platform.deepseek.com/) (free tier works)

### 1. Clone & configure

```bash
git clone https://github.com/YOUR_USERNAME/ai-music-studio.git
cd ai-music-studio
copy .env.example .env
# Edit .env → set your DEEPSEEK_API_KEY
```

### 2. Install ACE-Step (one-time)

```bash
cd vendor
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5
python -m venv .venv
.venv\Scripts\pip install -e .
# Model weights (~11 GB) auto-download on first launch
```

### 3. Install API dependencies

```bash
cd services\api
pip install -r requirements.txt
```

### 4. Launch

```
start.bat
```

The launcher handles everything:
- Downloads model weights if missing (~11 GB, one-time)
- Starts ACE-Step API server (127.0.0.1:8765) with GPU model loading
- Starts FastAPI backend (127.0.0.1:8000)
- Starts Next.js frontend (localhost:3000)
- Opens your browser to the studio

**First launch:** model initialization takes 2–5 minutes (GPU model loading). Subsequent launches are ~1–2 minutes.

## Architecture

```
Browser (localhost:3000)
    │  Next.js + Tailwind CSS
    │  /api/* → Next.js proxy → FastAPI
    ▼
FastAPI (127.0.0.1:8000)
    ├── POST /arrange  ──→ DeepSeek API (cloud, optional)
    ├── POST /generate ──→ Worker queue ──→ ACE-Step API (127.0.0.1:8765)
    ├── GET  /jobs/{id} ──→ SQLite (data/app.db)
    ├── GET  /library   ──→ SQLite
    └── GET  /audio/*   ──→ Local disk (outputs/)
    ▼
ACE-Step 1.5 API (127.0.0.1:8765)
    │  PyTorch + CUDA
    │  acestep-v15-turbo (DiT) + VAE + 5Hz-LM
    ▼
GPU Inference → WAV audio → outputs/
```

All services bind `127.0.0.1` only. No ports are exposed to the network.

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/arrange` | DeepSeek arrangement (prompt → structured lyrics + style) |
| `POST` | `/generate` | Submit music generation job |
| `GET` | `/jobs/{id}` | Query job status and result |
| `GET` | `/library` | List all songs (`?status=` filter) |
| `GET` | `/audio/{filename}` | Serve generated audio file |

### POST /arrange

```json
// Request
{ "prompt": "一首关于星空的治愈系国风歌曲，女声，钢琴", "lyrics": "" }

// Response
{
  "title": "星空下的约定",
  "language": "zh",
  "lyrics": "[Intro]\n...\n[Verse]\n星光洒落在窗前\n...\n[Chorus]\n...\n[Outro]",
  "style": {
    "genres": ["chinese-style", "pop"],
    "moods": ["healing", "dreamy"],
    "instruments": ["piano", "guzheng"],
    "vocals": ["female"],
    "tempo": "80-100"
  },
  "style_prompt": "chinese-style, pop, healing, dreamy, piano, guzheng, female vocals, 80-100 BPM"
}
```

### POST /generate

```json
// Request
{
  "title": "My Song",
  "lyrics": "[Verse]\n...",
  "style_prompt": "pop, piano, female vocals, 100-120 BPM",
  "instrumental": false
}

// Response
{ "id": "abc123", "status": "queued" }
```

### GET /jobs/{id}

```json
{
  "id": "abc123",
  "status": "succeeded",
  "title": "My Song",
  "audio_path": "outputs/abc123.wav",
  "created_at": "2026-07-15T17:00:00"
}
```

## Project Structure

```
ai-music-studio/
├── README.md
├── .env.example                     # Environment template
├── start.bat                        # One-click launcher
├── apps/
│   └── web/                         # Next.js frontend
│       ├── src/
│       │   ├── app/
│       │   │   ├── page.tsx         # Home
│       │   │   ├── studio/page.tsx  # Arrange room
│       │   │   ├── create/page.tsx  # Compose page
│       │   │   └── library/page.tsx # Music library
│       │   ├── components/
│       │   │   ├── Navbar.tsx
│       │   │   ├── StylePicker.tsx   # Genre/mood/instrument selector
│       │   │   └── StructureTags.tsx # Song section tag toolbar
│       │   └── lib/
│       │       └── api.ts           # API client
│       ├── next.config.js           # Proxy /api/* → FastAPI
│       └── tailwind.config.ts
├── services/
│   └── api/                         # FastAPI backend
│       ├── app/
│       │   ├── main.py              # App entry, lifespan
│       │   ├── config.py            # pydantic-settings
│       │   ├── db.py                # SQLAlchemy + SQLite
│       │   ├── models.py            # Job ORM
│       │   ├── schemas.py           # Request/response models
│       │   ├── routers/             # API route modules
│       │   ├── services/
│       │   │   ├── deepseek_arrange.py  # LLM arrangement
│       │   │   ├── acestep_runner.py    # ACE-Step interface
│       │   │   └── queue.py             # Async job worker
│       │   └── prompts/
│       │       └── arrange_system.txt   # DeepSeek system prompt
│       └── tests/                   # pytest (12 tests)
├── scripts/
│   └── download_models.py           # Model weight downloader
├── data/                            # SQLite database (gitignored)
├── outputs/                         # Generated audio (gitignored)
├── vendor/
│   └── ACE-Step-1.5/                # ACE-Step installation
└── docs/
    └── devlog.md
```

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Frontend | Next.js 14 + TypeScript + Tailwind CSS | Dark theme, responsive |
| API Gateway | FastAPI + Python | Async, pydantic-settings |
| Music Model | ACE-Step 1.5 (MIT) | DiT-based, GPU inference |
| LLM Arranger | DeepSeek API | Only cloud dependency |
| Database | SQLite (SQLAlchemy) | Zero-config local storage |
| Audio | WAV 48 kHz | FLAC/MP3 via ACE-Step config |

## Configuration

All settings in `.env`:

```env
# DeepSeek API (required for /arrange)
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com

# ACE-Step
ACESTEP_BACKEND=api              # "api" (recommended) or "subprocess"
ACESTEP_API_URL=http://127.0.0.1:8765
ACESTEP_DIR=G:/AI_Music/vendor/ACE-Step-1.5

# GPU optimization (auto-detected from GPU tier, override if needed)
# ACESTEP_QUANTIZATION=int8_weight_only   # int8_weight_only | fp8_weight_only | w8a8_dynamic | none
# ACESTEP_OFFLOAD_TO_CPU=true            # Enable CPU offload for VAE/text encoder
# ACESTEP_TMPDIR=G:/AI_Music/outputs     # Generated audio output directory

# Paths
OUTPUT_DIR=G:/AI_Music/outputs
DATABASE_URL=sqlite:///G:/AI_Music/data/app.db

# Network
API_HOST=127.0.0.1
API_PORT=8000
```

### GPU optimization for 8 GB cards

On GPUs with 8 GB VRAM, the API server automatically applies INT8 weight-only quantization to the DiT model, reducing its memory footprint from ~7 GB to ~3.5 GB. This is auto-detected from your GPU tier and requires no configuration.

Override with `ACESTEP_QUANTIZATION`:
- `int8_weight_only` — ~50% VRAM savings, negligible quality loss (default for ≤12 GB)
- `fp8_weight_only` — ~50% VRAM savings, slightly better precision (requires Ada/Hopper GPU)
- `w8a8_dynamic` — more aggressive, quantizes activations too
- `none` / `null` — disable quantization

## Troubleshooting

### "No module named 'app'" on FastAPI startup
Run from the `services/api` directory: `cd services\api && python -m uvicorn app.main:app`

### Port 8000 already in use
Previous FastAPI instance didn't shut down. Run `taskkill /F /IM python.exe` or restart `start.bat` (v2.0+ auto-cleans ports).

### ACE-Step model download fails
China mainland: ModelScope mirror is auto-selected. If both HuggingFace and ModelScope fail, set proxy:
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### CUDA out of memory
Your GPU has <8 GB VRAM. Enable CPU offloading:
```bash
set ACESTEP_OFFLOAD_TO_CPU=true
```
Generation will be slower but functional.

### Generated audio has noise/artifacts
This is a known limitation of current DiT-based music models. See [discussion below](#ai-voice-quality).

## AI Voice Quality

The ACE-Step 1.5 turbo model produces acceptable instrumental quality but vocals can exhibit:

- **Sibilance artifacts** — high-frequency "static" on "s", "sh", "ch" sounds
- **Pitch instability** — slight wavering on sustained notes
- **Timbre inconsistency** — vocal character shifts between song sections
- **Muffled consonants** — "b", "d", "g" can lack clarity in dense mixes

### Mitigation strategies (no retraining needed)

1. **Style prompt engineering** — Add `"clear vocals, crisp, high quality"` to the prompt
2. **Reduce instrumental density** — Fewer concurrent instruments in the prompt → cleaner vocals
3. **Post-processing** — Run output through:
   - EQ cut at 4–8 kHz to reduce sibilance
   - Gentle compression (2:1 ratio) to even out dynamics
   - A de-esser plugin for vocal sibilance (e.g., `sox` CLI: `sox input.wav output.wav deharsh 0.7`)
4. **Use the SFT model** — `acestep-v15-sft` has better vocal quality than `turbo` at the cost of slower inference (set `ACESTEP_CONFIG_PATH=acestep-v15-sft`)

### Future improvements

- Fine-tuning with high-quality Chinese vocal datasets
- LoRA adapters for specific vocal styles
- Two-stage generation: instrument track + vocal track separately

## Roadmap

### v0.2 — Quality & Polish

- [ ] **SFT model support** — `acestep-v15-sft` for higher vocal quality (trade-off: 3× slower inference)
- [ ] **Custom voice profiles** — save and reuse vocal characteristics across songs (like Suno's Persona)
- [ ] **Stem separation** — split output into vocals + accompaniment for post-processing
- [ ] **Post-processing pipeline** — automatic de-esser, EQ normalization, loudness normalization (LUFS)
- [ ] **FLAC/MP3 output** — configurable output format with proper tagging

### v0.3 — Creative Tools

- [ ] **Voice cloning / LoRA adapters** — fine-tune on a reference vocal sample to reproduce a specific voice
- [ ] **Style library** — curated presets for common Chinese music genres (古风, 国风 R&B, 民谣, C-Pop, etc.)
- [ ] **Multi-language vocals** — Japanese, Korean, English vocal support
- [ ] **Cover mode** — upload a reference track, ACE-Step generates in-the-style-of arrangement
- [ ] **Inpainting / repainting** — re-generate specific song sections without touching the rest

### v0.4 — Ecosystem

- [ ] **Docker Compose one-click** — fully containerized setup including CUDA
- [ ] **Suno-API compatibility layer** — drop-in API replacement for tools that speak Suno's API
- [ ] **VST/AU plugin bridge** — route generated audio directly into DAW workflows
- [ ] **Training recipes** — documentation and scripts for LoRA fine-tuning on custom datasets

### Custom Voice / Persona (like Suno)

The goal for v0.2 is a "Voice Profile" system:

1. **Capture** — record or upload 30–60 seconds of target vocals
2. **Extract** — compute a vocal embedding using a speaker verification model (e.g. Resemblyzer / ECAPA-TDNN)
3. **Condition** — inject the embedding into ACE-Step's DiT cross-attention layers during inference, steering vocal timbre toward the target
4. **Reuse** — saved profiles persist locally, selectable from a dropdown in the Create page

This is technically feasible because ACE-Step's DiT backbone uses cross-attention for conditioning — the same mechanism that injects text prompts. Adding a speaker embedding is architecturally straightforward; the main work is training/fine-tuning the conditioning adapter.

For v0.3, we aim to go further with **LoRA-based voice cloning**: fine-tune a low-rank adapter on a longer reference sample (3–5 minutes of clean vocals) for higher-fidelity voice reproduction. LoRA adapters are small (~20–50 MB) and can be shared as `.safetensors` files.

---

## License

This project — MIT License.

ACE-Step 1.5 — MIT License. Model weights from [ACE-Step](https://github.com/ace-step/ACE-Step-1.5).

DeepSeek API — subject to [DeepSeek Terms of Service](https://platform.deepseek.com/).

---

**Built for creators who own their tools.**
