"""ACE-Step 1.5 runner interface.

Supports:
- api: call ACE-Step's REST API server (recommended)
- subprocess: call ACE-Step CLI via uv run (one-shot, no server needed)
"""

import json
import asyncio
from pathlib import Path

from app.config import settings

BACKEND = settings.acestep_backend
ACESTEP_DIR = settings.acestep_dir


def _ensure_output_dir():
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)


async def generate(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool = False,
) -> Path:
    _ensure_output_dir()

    if BACKEND == "subprocess":
        return await _generate_subprocess(lyrics, style_prompt, output_path, instrumental)
    elif BACKEND == "api":
        return await _generate_api(lyrics, style_prompt, output_path, instrumental)
    else:
        raise ValueError(f"Unknown ACESTEP_BACKEND: {BACKEND}")


async def _generate_subprocess(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool,
) -> Path:
    """Call ACE-Step via uv run acestep-api in one-shot mode.

    Writes a JSON config and pipes it through CLI.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write lyrics to temp file
    lyrics_path = output_dir / "lyrics_input.txt"
    lyrics_path.write_text(lyrics, encoding="utf-8")

    # Build ACE-Step config JSON
    config = {
        "prompt": style_prompt,
        "lyrics": lyrics,
        "instrumental": instrumental,
        "vocal_language": "zh",
        "task_type": "text2music",
        "thinking": False,
        "inference_steps": 25,
        "guidance_scale": 7.0,
        "audio_format": "wav",
    }

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
) -> Path:
    """Call ACE-Step's REST API server (acestep-api)."""
    import httpx

    api_url = settings.acestep_api_url

    payload = {
        "prompt": style_prompt,
        "lyrics": lyrics,
        "vocal_language": "zh",
        "task_type": "text2music",
        "thinking": False,
        "inference_steps": 25,
        "guidance_scale": 7.0,
        "audio_format": "wav",
    }
    if instrumental:
        payload["prompt"] += ", instrumental, no vocals"

    # 1. Submit task
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{api_url}/release_task", json=payload)
        resp.raise_for_status()
        task = resp.json()
        task_id = task["task_id"]

    # 2. Poll until done
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(120):  # ~10 min timeout
            await asyncio.sleep(5)
            resp = await client.post(
                f"{api_url}/query_result",
                json={"task_ids": [task_id]},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", {})
            if task_id in results:
                r = results[task_id]
                if r["status"] == "succeeded":
                    audio_path = r.get("audio_path", "")
                    break
                elif r["status"] == "failed":
                    raise RuntimeError(f"Generation failed: {r.get('error', '')}")
        else:
            raise TimeoutError("Generation timed out")

    # 3. Download audio
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(
            f"{api_url}/v1/audio",
            params={"path": audio_path},
        )
        resp.raise_for_status()
        Path(output_path).write_bytes(resp.content)

    return Path(output_path)
