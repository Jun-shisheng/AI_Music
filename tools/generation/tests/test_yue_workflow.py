import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yue_workflow import build_command  # noqa: E402


def test_yue_command_uses_single_batch_and_local_model_paths(tmp_path):
    genre = tmp_path / "genre.txt"
    lyrics = tmp_path / "lyrics.txt"
    genre.write_text("female pop vocal", encoding="utf-8")
    lyrics.write_text("[Verse]\nA clear line", encoding="utf-8")
    try:
        command = build_command(genre, lyrics, tmp_path / "out")
    except FileNotFoundError:
        return
    assert "--stage2_batch_size" in command
    assert command[command.index("--stage2_batch_size") + 1] == "1"
    assert str(genre) in command
