import json
import re

import httpx

from app.config import settings
from app.services.lyric_diversity import analyze_lyrics

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
    parts.append(
        "\n".join(
            [
                "正式创作标准：",
                "1. 默认创作人声完整歌曲，不写纯音乐，除非用户明确要求。",
                "2. 目标成歌时长约 3 分 30 秒，歌词需要足够支撑完整结构。",
                "3. 结构建议包含：[前奏] [主歌1] [预副歌1] [副歌1] [间奏] [主歌2] [预副歌2] [副歌2] [桥段] [最终副歌] [尾奏]。",
                "4. 主歌2必须推进叙事，不要只是改写主歌1；副歌2/最终副歌可以保留核心 hook，但需要有情绪递进或少量词句变化。",
                "5. 歌词避免大段机械重复：非副歌 unique 要高；禁止连续三行以上相同；桥段要有新信息。",
                "6. 唱腔需要偏真人流行演唱：允许拉长音、转音、气口、弱起、尾音处理和少量即兴和声，不要逐字机械咬字。",
                "7. 高音段要写成连贯可唱的乐句，避免突然断裂、尖锐机器噪声、初音未来式电子人声质感。",
                "8. style_prompt 需包含编曲动态：如 rising pre-chorus, contrasted bridge, layered final chorus, instrument entries。",
            ]
        )
    )
    return "\n\n".join(parts)


def _rewrite_prompt_for_diversity(requirement: str, previous: dict, issues: list[str], language: str) -> str:
    return "\n\n".join(
        [
            "上一次编排未通过「反疲劳重复」检查，请在保留主题与风格的前提下重写歌词。",
            f"原要求：{requirement}",
            f"语言：{'中文' if language == 'zh' else '英文'}",
            "问题清单：\n- " + "\n- ".join(issues),
            "上一版歌词：\n" + str(previous.get("lyrics", "")),
            "上一版 style_prompt：" + str(previous.get("style_prompt", "")),
            "请输出完整 JSON（含 title/language/lyrics/style/style_prompt），修复上述问题，副歌 hook 可保留但主歌必须有推进。",
        ]
    )


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
    max_retries: int = 2,
) -> dict:
    api_key = settings.deepseek_api_key
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    system_prompt = _load_system_prompt()
    user_prompt = _build_user_prompt(requirement, lyrics, language)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    parsed: dict | None = None

    for attempt in range(max_retries + 1):
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
                diversity = analyze_lyrics(parsed.get("lyrics", ""))
                if diversity.passes:
                    parsed["lyric_diversity"] = {
                        "passes": True,
                        "unique_ratio": diversity.unique_ratio,
                        "verse_similarity": diversity.verse_similarity,
                    }
                    return parsed
                # One targeted rewrite for fatigue issues
                if attempt < max_retries:
                    user_prompt = _rewrite_prompt_for_diversity(
                        requirement, parsed, diversity.issues, language
                    )
                    last_error = ValueError(diversity.summary())
                    continue
                raise ValueError(f"Lyric diversity check failed after rewrite: {diversity.summary()}")
        except (json.JSONDecodeError, KeyError, httpx.HTTPError, ValueError) as e:
            last_error = e
            if attempt == max_retries:
                raise
            continue

    if last_error:
        raise last_error
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
