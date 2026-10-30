"""ACE-Step 1.5 runner interface.

Supports:
- api: call ACE-Step's REST API server (recommended)
- subprocess: call ACE-Step CLI via uv run (one-shot, no server needed)
"""

import json
import asyncio
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

from app.config import settings
from app.services.music_text import detect_vocal_language, prepare_generation_lyrics

BACKEND = settings.acestep_backend
ACESTEP_DIR = settings.acestep_dir
_lora_loaded_path: str | None = None


def _ensure_output_dir():
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)


async def _ensure_lora_loaded() -> None:
    """Load ACE-Step LoRA via HTTP API once per process when configured."""
    global _lora_loaded_path
    lora_path = (settings.acestep_lora_path or "").strip()
    if not lora_path:
        return
    if BACKEND != "api":
        return
    if _lora_loaded_path == lora_path:
        return
    import httpx

    api_url = settings.acestep_api_url
    payload: dict = {"lora_path": lora_path}
    if settings.acestep_lora_adapter_name:
        payload["adapter_name"] = settings.acestep_lora_adapter_name
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{api_url}/v1/lora/load", json=payload)
            resp.raise_for_status()
            await client.post(
                f"{api_url}/v1/lora/scale",
                json={
                    "scale": float(settings.acestep_lora_scale),
                    "adapter_name": settings.acestep_lora_adapter_name or None,
                },
            )
            await client.post(f"{api_url}/v1/lora/toggle", json={"use_lora": True})
        _lora_loaded_path = lora_path
    except Exception as exc:  # noqa: BLE001
        print(f"[acestep] LoRA load skipped/failed ({lora_path}): {exc}")


def _quality_prompt(style_prompt: str, instrumental: bool, duration_seconds: int | None) -> str:
    parts = [style_prompt.strip()]
    if instrumental:
        parts.extend([
            "instrumental",
            "no vocals",
            "clean mix",
            "no noise",
            "no distortion",
        ])
    else:
        parts.extend([
            "complete full-length song",
            "realistic human lead vocal tone",
            "warm natural female vocal when applicable",
            "instrumental intro before the first verse",
            "lead vocal enters at the first verse",
            "sing only the provided lyric lines",
            "no humming intro",
            "no wordless vocal filler",
            "no la la or ooh ah ad-lib filler",
            "natural pronunciation",
            "smooth connected syllables",
            "stable high notes",
            "controlled sustained notes inside lyric phrases",
            "subtle tasteful vocal variation",
            "dynamic phrasing",
            "low artifacts",
            "no harsh AI noise",
            "no sudden machine noise",
            "no robotic synthetic timbre",
            "not syllable-by-syllable chanting",
            "no distortion",
            "varied arrangement",
            "dynamic section contrast",
            "instrument entries and exits",
            "rising pre-chorus",
            "contrasted bridge",
            "layered final chorus",
            "verse pre-chorus chorus bridge final chorus",
        ])
    if duration_seconds and duration_seconds >= 180:
        parts.append("full song structure with evolving melody and instrumentation")
    return ", ".join(part for part in parts if part)


async def generate(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool = False,
    duration_seconds: int | None = None,
    loopable: bool = False,
) -> Path:
    _ensure_output_dir()
    await _ensure_lora_loaded()

    if BACKEND == "subprocess":
        return await _generate_subprocess(
            lyrics, style_prompt, output_path, instrumental, duration_seconds, loopable
        )
    elif BACKEND == "api":
        return await _generate_api(
            lyrics, style_prompt, output_path, instrumental, duration_seconds, loopable
        )
    else:
        raise ValueError(f"Unknown ACESTEP_BACKEND: {BACKEND}")


async def _generate_subprocess(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool,
    duration_seconds: int | None,
    loopable: bool,
) -> Path:
    """Call ACE-Step via uv run acestep-api in one-shot mode.

    Writes a JSON config and pipes it through CLI.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    lyrics = prepare_generation_lyrics(lyrics) if not instrumental else lyrics

    # Write lyrics to temp file
    lyrics_path = output_dir / "lyrics_input.txt"
    lyrics_path.write_text(lyrics, encoding="utf-8")

    prompt = _quality_prompt(style_prompt, instrumental, duration_seconds)
    if loopable:
        prompt = f"{prompt}, seamless loop, loopable game background music"

    # Build ACE-Step config JSON
    config = {
        "prompt": prompt,
        "lyrics": lyrics or ("[Instrumental]" if instrumental else ""),
        "instrumental": instrumental,
        "vocal_language": detect_vocal_language(lyrics, style_prompt),
        "task_type": "text2music",
        "thinking": False,
        "inference_steps": settings.acestep_inference_steps,
        "guidance_scale": settings.acestep_guidance_scale,
        "audio_format": "wav",
    }
    if duration_seconds:
        config["audio_duration"] = duration_seconds

    config_path = output_dir / "acestep_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    # Use uv run to call ACE-Step's generation directly
    cmd = [
        "uv", "run", "--no-sync",
        "python", "-c", f"""
import json, sys
sys.path.insert(0, '{ACESTEP_DIR}')
from acestep.api.job_blocking_generation import run_blocking_generate
from acestep.api.http.release_task_models import GenerateMusicRequest

config = json.loads(open('{config_path}').read())
req = GenerateMusicRequest(**config)
audio, _ = run_blocking_generate(req)
with open('{output_path}', 'wb') as f:
    f.write(audio)
print("DONE")
""",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=ACESTEP_DIR,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ACE-Step failed: {stderr.decode()}")

    out = Path(output_path)
    if not out.exists():
        raise FileNotFoundError(f"No output at {output_path}")
    return out


async def _generate_api(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool,
    duration_seconds: int | None,
    loopable: bool,
) -> Path:
    """Call ACE-Step's REST API server (acestep-api).

    Flow: submit task → poll for completion → download audio.
    """
    import httpx

    api_url = settings.acestep_api_url
    lyrics = prepare_generation_lyrics(lyrics) if not instrumental else lyrics
    prompt = _quality_prompt(style_prompt, instrumental, duration_seconds)
    if loopable:
        prompt += ", seamless loop, loopable game background music, clean loop point"

    payload = {
        "prompt": prompt,
        "lyrics": lyrics or ("[Instrumental]" if instrumental else ""),
        "vocal_language": detect_vocal_language(lyrics, style_prompt),
        "task_type": "text2music",
        "thinking": False,
        "inference_steps": settings.acestep_inference_steps,
        "guidance_scale": settings.acestep_guidance_scale,
        "audio_format": "wav",
    }
    if duration_seconds:
        payload["audio_duration"] = duration_seconds

    # 1. Submit task
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{api_url}/release_task", json=payload)
        resp.raise_for_status()
        body = resp.json()
        task_id = body["data"]["task_id"]

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
