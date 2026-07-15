import json
import re

import httpx

from app.config import settings

SYSTEM_PROMPT_PATH = "app/prompts/arrange_system.txt"

_system_prompt: str | None = None


def _load_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        try:
            _system_prompt = open(SYSTEM_PROMPT_PATH, encoding="utf-8").read()
        except FileNotFoundError:
            _system_prompt = ""
    return _system_prompt


def _build_user_prompt(requirement: str, lyrics: str = "", language: str = "zh") -> str:
    parts = [f"要求：{requirement}"]
    if lyrics:
        parts.append(f"用户提供的歌词（可调整结构标签）：\n{lyrics}")
    parts.append(f"语言：{'中文' if language == 'zh' else '英文'}")
    return "\n\n".join(parts)


def _extract_json(raw: str) -> str:
    """Extract JSON from model output that might be wrapped in markdown fences."""
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        return match.group(0)
    return raw


async def arrange(
    requirement: str,
    lyrics: str = "",
    language: str = "zh",
    max_retries: int = 1,
) -> dict:
    api_key = settings.deepseek_api_key
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    system_prompt = _load_system_prompt()
    user_prompt = _build_user_prompt(requirement, lyrics, language)

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{settings.deepseek_base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(_extract_json(content))
                _validate_arrangement(parsed)
                return parsed
        except (json.JSONDecodeError, KeyError, httpx.HTTPError) as e:
            if attempt == max_retries:
                raise
            continue

    raise RuntimeError("Unexpected: all retries exhausted")


def _validate_arrangement(data: dict):
    required_keys = ["title", "language", "lyrics", "style", "style_prompt"]
    style_keys = ["genres", "moods", "instruments", "vocals", "tempo"]

    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    style = data.get("style", {})
    missing_style = [k for k in style_keys if k not in style]
    if missing_style:
        raise ValueError(f"Missing style fields: {missing_style}")
