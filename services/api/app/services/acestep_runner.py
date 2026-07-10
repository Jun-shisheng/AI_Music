"""ACE-Step 1.5 runner interface.

Supports multiple backends:
- subprocess: call ACE-Step CLI directly (simplest, no code dependency)
- api: call ACE-Step's built-in REST API server
- import: import ACE-Step pipeline as a library (tightest integration)

Set ACESTEP_BACKEND env var to choose: subprocess (default), api, or import.
"""

import os
import subprocess
import json
import tempfile
from pathlib import Path

from app.config import settings

BACKEND = os.getenv("ACESTEP_BACKEND", "subprocess")


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
    elif BACKEND == "import":
        return await _generate_import(lyrics, style_prompt, output_path, instrumental)
    else:
        raise ValueError(f"Unknown ACESTEP_BACKEND: {BACKEND}")


async def _generate_subprocess(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool,
) -> Path:
    """Call ACE-Step via its CLI. Requires ACE-Step installed and on PATH."""
    lyrics_path = Path(output_path).with_suffix(".txt")
    lyrics_path.write_text(lyrics, encoding="utf-8")

    cmd = [
        "python", "-m", "acestep.acestep_v15_pipeline",
        "--lyrics", str(lyrics_path),
        "--style_prompt", style_prompt,
        "--output", str(output_path),
    ]
    if instrumental:
        cmd.append("--instrumental")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ACE-Step subprocess failed: {result.stderr}")

    out = Path(output_path)
    if not out.exists():
        raise FileNotFoundError(f"ACE-Step did not produce output at {output_path}")
    return out


async def _generate_api(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool,
) -> Path:
    """Call ACE-Step's built-in REST API server."""
    import httpx

    api_url = os.getenv("ACESTEP_API_URL", "http://127.0.0.1:7860")
    payload = {
        "lyrics": lyrics,
        "style_prompt": style_prompt,
        "instrumental": instrumental,
    }
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(f"{api_url}/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        audio_bytes = bytes(data["audio"])

    Path(output_path).write_bytes(audio_bytes)
    return Path(output_path)


async def _generate_import(
    lyrics: str,
    style_prompt: str,
    output_path: str,
    instrumental: bool,
) -> Path:
    """Import ACE-Step pipeline directly as a Python library.

    Requires: pip install -e path/to/ACE-Step-1.5
    """
    try:
        from acestep.acestep_v15_pipeline import AceStepPipeline
    except ImportError:
        raise ImportError(
            "ACE-Step is not installed as a library. "
            "Install with: pip install -e /path/to/ACE-Step-1.5"
        )

    model_dir = settings.acestep_model_dir
    pipe = AceStepPipeline.from_pretrained(model_dir)
    output = pipe.generate(
        lyrics=lyrics,
        style_prompt=style_prompt,
        instrumental=instrumental,
    )
    output.save(output_path)
    return Path(output_path)
