# Front end ↔ back end contract

Everything the UI sends and everything it gets back, in one place, so the two
sides can be changed without guessing. The front end is `apps/web`; the back end
is `services/api`. The authoritative definitions are
`services/api/app/schemas.py` (payloads) and `apps/web/src/lib/api.ts` (the
client). This document is the contract between them, and the tests under
`services/api/tests/` enforce the parts that can drift silently.

## Wiring

```
browser  ──►  Next.js dev server (localhost:3000)
                   │  rewrites /api/:path*  ──►  http://127.0.0.1:8000/:path*
                   ▼
              FastAPI (127.0.0.1:8000)
                   │  POST /arrange ──► LLM (optional, cloud)
                   │  POST /generate ─► in-process queue ──► local model
                   └─ GET /audio/... ─► files on disk (OUTPUT_DIR)
```

- The browser never talks to port 8000 directly: `apps/web/next.config.js`
  rewrites `/api/*` and strips the prefix. For a custom backend port, pass
  `--port` to uvicorn and set `API_PROXY_TARGET=http://127.0.0.1:<port>` when
  starting Next dev or building Next for production. `.env`'s `API_PORT` alone
  does not configure either CLI. `start.bat` uses ports 8765, 8000 and 3000 and
  exits on conflicts without killing the processes that own those ports.
- CORS is restricted to `http://localhost:3000` (`app/main.py`). Serving the UI
  from another origin needs that list updated.
- Both services bind `127.0.0.1` only; nothing is exposed to the network.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness. `{"status": "ok"}` |
| `POST` | `/arrange` | LLM turns an idea into structured lyrics + style |
| `POST` | `/generate` | Queue a generation job |
| `GET` | `/jobs/{id}` | Job status, and the result when finished |
| `GET` | `/library` | All jobs, newest first (`?status=` filters) |
| `GET` | `/audio/{filename}` | Serves a generated `.wav` that is a direct child of `OUTPUT_DIR`; anything else is a 404 |

### POST /arrange

```jsonc
// request
{ "prompt": "一首关于星空的治愈系国风歌曲，女声，钢琴", "lyrics": "" }

// 200
{
  "title": "星空下的约定",
  "language": "zh",
  "lyrics": "[Intro]\n…\n[Verse]\n…\n[Chorus]\n…\n[Outro]",
  "style": {
    "genres": ["chinese-style", "pop"], "moods": ["healing", "dreamy"],
    "instruments": ["piano", "guzheng"], "vocals": ["female"], "tempo": "80-100"
  },
  "style_prompt": "chinese-style, pop, healing, dreamy, piano, guzheng, female vocals, 80-100 BPM"
}
```

`style_prompt` is the string everything downstream uses. Requires
`DEEPSEEK_API_KEY`; without it the route returns an error and the UI shows the
message — **lyrics can always be typed by hand instead**, which is what the
Generate step needs.

### POST /generate

```jsonc
// request  (every field except lyrics/style_prompt has a default)
{
  "title": "My Song",
  "lyrics": "[Verse]\n…",
  "style_prompt": "pop, female vocals, 100-120 BPM",
  "instrumental": false,
  "generation_type": "song",        // song | bgm_loop | sfx
  "duration_seconds": 120,          // optional, 10-600
  "loopable": false,
  "variation_label": "take A"       // optional, ≤80 chars, for your own reference
}

// 201
{ "id": "abc123", "status": "queued" }
```

Validation is by `GenerateRequest` (pydantic): an unknown `generation_type`, a
duration outside 10-600, or a `variation_label` over 80 characters is a **422**
before any work starts.

### `generation_type` — what the three types actually change

The type is not cosmetic: `acestep_runner.resolve_request()` maps it onto the
request that goes to the model, and `test_generation_type.py` pins the mapping.

| type | vocals | duration | prompt additions |
|---|---|---|---|
| `song` | allowed (`instrumental: true` for none) | as asked | — |
| `bgm_loop` | never | as asked | `game background music, instrumental, no vocals, seamless loop, loopable, clean loop point` |
| `sfx` | never | defaults to **10 s** if not set | `sound effect, no music, no vocals` |

When the resolved request is instrumental (`instrumental: true`, `bgm_loop`,
or `sfx`), lyrics are replaced with `[Instrumental]` before storage and before
calling the backend, even if the caller supplied vocal lyrics. The job and
library therefore show the effective lyrics used for generation.

### GET /jobs/{id}

```jsonc
{
  "id": "abc123", "title": "My Song",
  "lyrics": "…", "style_prompt": "…",
  "instrumental": false, "generation_type": "song",
  "duration_seconds": 120, "loopable": false, "variation_label": null,
  "status": "queued | running | succeeded | failed",
  "audio_path": "outputs/abc123.wav",   // path relative to the API's cwd
  "audio_url": "/audio/abc123.wav",     // what the UI should play
  "error": null,                        // populated when status == "failed"
  "created_at": "2026-07-15T17:00:00", "updated_at": "2026-07-15T17:02:11"
}
```

`status` is the only field the UI polls on; a job that fails carries the
generator's message in `error` rather than an HTTP error, so a failed song is
visible in the library instead of vanishing.

Run the API with a **single worker process**. Queue recovery at startup marks
interrupted `running` jobs as failed and restores `queued` jobs from the database.
Within that process, each queued job is claimed once; duplicate queue entries
cannot restart running or finished jobs. Stopping the worker records an
interrupted job as failed immediately. A missing or empty output file is a failed
generation. Restart recovery does not resume or cancel work in the separate
ACE-Step server; multi-process queue coordination is not implemented.

### GET /library

Same objects as `/jobs/{id}`, newest first. `?status=succeeded` filters.

## What must be installed for generation to work

`POST /generate` queues work that runs the local model. None of this is in the
repository, and each missing piece surfaces as a `failed` job with an `error`
message rather than a crash:

1. **The model checkout** at `vendor/ACE-Step-1.5` (or `ACESTEP_DIR` in `.env`).
   `scripts/download_models.py` and `start.bat` fetch ~10 GB of weights into
   `vendor/ACE-Step-1.5/checkpoints`.
2. **A CUDA PyTorch** in that checkout's venv — the runner talks to ACE-Step's
   REST API (`ACESTEP_BACKEND=api`, needs the server on `ACESTEP_API_URL`).
   `ACESTEP_BACKEND=subprocess` is unsupported and returns an explicit error.
3. **`OUTPUT_DIR`** must be writable; generated WAVs land there and are served
   back through `/audio/{filename}`.

`diction_qa.py` (in `tools/generation/`) additionally wants a
`faster-whisper` model on disk, passed with `--model`.

## Changing the contract

Anything under `services/api/app/schemas.py` is the contract: add the field
there, in the ORM model (`app/models.py`), in the TS interfaces
(`apps/web/src/lib/api.ts`), and — if it changes what the model is asked for —
in `resolve_request()`, with a test.

Tables are created by `Base.metadata.create_all`, and the four generation
columns are back-filled onto an existing `jobs` table by
`db._ensure_job_columns()`. **Any other new column needs its own line in that
function**, or an old `data/app.db` will fail inserts with "no such column" —
`create_all` does not alter an existing table.

The sqlite file's directory is created at import time
(`db.ensure_sqlite_dir()`): the default URL is relative, and sqlite will not
create a missing directory — without it a fresh clone dies during startup with
"unable to open database file".
