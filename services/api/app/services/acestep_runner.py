"""ACE-Step 1.5 runner interface.

The supported backend is `api`: talk to the ACE-Step REST server that
`start.bat` launches on ACESTEP_API_URL. The old one-shot `subprocess` path is
kept only as a named seam for tests -- ACE-Step 1.5's blocking generator needs
the server's app_state and handlers, so it cannot be called from a bare
`python -c`.
"""

import json
import asyncio
from pathlib import Path

from app.config import settings

BACKEND = settings.acestep_backend
ACESTEP_DIR = settings.acestep_dir

GENERATION_TYPES = ("song", "bgm_loop", "sfx")

# A sound effect has no natural length; a song's does. 10 s is the API's floor.
SFX_DEFAULT_SECONDS = 10

_TYPE_PROMPT_BITS: dict[str, list[str]] = {
    "song": [],
    "bgm_loop": ["game background music", "instrumental", "no vocals"],
    "sfx": ["sound effect", "no music", "no vocals"],
}
_LOOP_PROMPT_BITS = ["seamless loop", "loopable", "clean loop point"]


def resolve_request(
    generation_type: str,
    instrumental: bool,
    duration_seconds: int | None,
    loopable: bool,
) -> dict:
    """Turn the API's generation type into what the model is actually asked for.

    `generation_type` is part of the request contract the front end sends
    (`song` / `bgm_loop` / `sfx`), but it used to be stored on the job and never
    reach the generator -- choosing "sound effect" produced exactly what
    "song" produced. One function decides the mapping so the contract cannot
    drift again, and so it can be tested without a model:

      song      vocals allowed; the lyrics decide everything else
      bgm_loop  instrumental, seamless, framed as looping background music
      sfx       instrumental and explicitly non-musical, short by default
    """
    if generation_type not in GENERATION_TYPES:
        raise ValueError(
            f"Unknown generation_type {generation_type!r}; expected one of "
            f"{', '.join(GENERATION_TYPES)}")
    instrumental = instrumental or generation_type != "song"
    loopable = loopable or generation_type == "bgm_loop"
    if generation_type == "sfx" and not duration_seconds:
        duration_seconds = SFX_DEFAULT_SECONDS
    prompt_bits = list(_TYPE_PROMPT_BITS[generation_type])
    if generation_type == "song" and instrumental:
        prompt_bits += ["instrumental", "no vocals"]
    if loopable:
        prompt_bits += _LOOP_PROMPT_BITS
    return {"instrumental": instrumental, "duration_seconds": duration_seconds,
            "loopable": loopable, "prompt_bits": prompt_bits}


def _ensure_output_dir():
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)


async def generate(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool = False,
    duration_seconds: int | None = None,
    loopable: bool = False,
    generation_type: str = "song",
) -> Path:
    _ensure_output_dir()
    req = resolve_request(generation_type, instrumental, duration_seconds, loopable)
    # Instrumental mode must also replace stale lyrics from an earlier vocal
    # request; prompt additions alone do not stop the model from singing them.
    lyrics = "[Instrumental]" if req["instrumental"] else lyrics

    if BACKEND == "subprocess":
        return await _generate_subprocess(
            lyrics, style_prompt, output_path, req["instrumental"],
            req["duration_seconds"], req["prompt_bits"]
        )
    elif BACKEND == "api":
        return await _generate_api(
            lyrics, style_prompt, output_path, req["instrumental"],
            req["duration_seconds"], req["prompt_bits"]
        )
    else:
        raise ValueError(f"Unknown ACESTEP_BACKEND: {BACKEND}")


async def _generate_subprocess(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool,
    duration_seconds: int | None,
    prompt_bits: list[str],
) -> Path:
    """Unsupported one-shot path, kept as a named seam for tests.

    ACE-Step 1.5's `run_blocking_generate()` is keyword-only and needs ~21
    runtime objects owned by the API server (app_state, model store, LLM
    handler, generation callbacks). A fresh `python -c` process cannot supply
    them, which is why the old implementation could never produce audio. Use
    the REST backend instead.
    """
    raise RuntimeError(
        "ACESTEP_BACKEND=subprocess is not supported by ACE-Step 1.5. "
        "Start the ACE-Step API server (start.bat does this) and set "
        "ACESTEP_BACKEND=api."
    )


async def _generate_api(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool,
    duration_seconds: int | None,
    prompt_bits: list[str],
) -> Path:
    """Call ACE-Step's REST API server (acestep-api).

    Flow: submit task → poll for completion → download audio.
    """
    import httpx

    api_url = settings.acestep_api_url
    prompt = ", ".join(p for p in [style_prompt, *prompt_bits] if p)

    payload = {
        "prompt": prompt,
        "lyrics": "[Instrumental]" if instrumental else lyrics,
        "vocal_language": "zh",
        "task_type": "text2music",
        "thinking": False,
        "inference_steps": 8,
        "guidance_scale": 7.0,
        "audio_format": "wav",
    }
    if duration_seconds:
        payload["audio_duration"] = duration_seconds

    audio_rel = ""
    try:
        # 1. Submit task
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{api_url}/release_task", json=payload)
            resp.raise_for_status()
            body = resp.json()
            task_id = body["data"]["task_id"]
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot reach the ACE-Step API server at {api_url}. "
            "Start it with start.bat, or set ACESTEP_API_URL to the running server."
        ) from exc

    # 2. Poll until done (status: 0=running, 1=succeeded, 2=failed)
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(120):  # ~10 min timeout
            await asyncio.sleep(5)
            resp = await client.post(
                f"{api_url}/query_result",
                json={"task_id_list": json.dumps([task_id])},
            )
            resp.raise_for_status()
            body = resp.json()
            items = body.get("data", [])
            for item in items:
                if item.get("task_id") != task_id:
                    continue
                status = item.get("status")
                if status == 1:  # succeeded
                    result_str = item.get("result", "[]")
                    results = json.loads(result_str) if isinstance(result_str, str) else result_str
                    audio_rel = results[0].get("file", "") if results else ""
                    break
                elif status == 2:  # failed
                    raise RuntimeError(f"Generation failed: {item.get('progress_text', '')}")
                audio_rel = ""
            else:
                continue
            break  # got a terminal status
        else:
            raise TimeoutError("Generation timed out (10 min)")

    if not audio_rel:
        raise RuntimeError("No audio file in generation result")

    # 3. Download audio
    # ACE-Step returns file as either:
    #   - URL: "/v1/audio?path=<encoded_path>"
    #   - or raw filesystem path
    async with httpx.AsyncClient(timeout=120) as client:
        if audio_rel.startswith("/v1/audio"):
            resp = await client.get(f"{api_url}{audio_rel}")
        else:
            resp = await client.get(f"{api_url}/v1/audio", params={"path": audio_rel})
        resp.raise_for_status()
        Path(output_path).write_bytes(resp.content)

    return Path(output_path)
