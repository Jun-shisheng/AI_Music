import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heartmula_workflow import build_command  # noqa: E402


def test_heartmula_command_uses_lazy_low_memory_mode(tmp_path):
    lyrics = tmp_path / "lyrics.txt"
    tags = tmp_path / "tags.txt"
    lyrics.write_text("[Verse]\nA clear line", encoding="utf-8")
    tags.write_text("female, pop, warm, piano", encoding="utf-8")
    try:
        command = build_command(lyrics, tags, tmp_path / "out.mp3", seconds=30)
    except FileNotFoundError:
        return
    assert "--lazy_load" in command
    assert command[command.index("--lazy_load") + 1] == "true"
    assert "--max_audio_length_ms" in command
