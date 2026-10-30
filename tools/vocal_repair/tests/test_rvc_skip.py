import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from rvc_convert import convert_vocals, discover_cover_vision, is_rvc_configured  # noqa: E402


def test_rvc_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("RVC_DISABLE_AUTO", "1")
    monkeypatch.delenv("RVC_ENABLED", raising=False)
    monkeypatch.delenv("RVC_MODEL_PATH", raising=False)
    monkeypatch.delenv("RVC_INFER_CMD", raising=False)
    assert is_rvc_configured(enabled=False) is False
    result = convert_vocals(Path("in.wav"), Path("out.wav"), enabled=False)
    assert result.status == "skipped_not_configured"


def test_cover_vision_auto_detect_when_present(monkeypatch):
    monkeypatch.delenv("RVC_DISABLE_AUTO", raising=False)
    monkeypatch.delenv("RVC_ENABLED", raising=False)
    monkeypatch.delenv("RVC_INFER_CMD", raising=False)
    preset = discover_cover_vision()
    if preset is None:
        # Machine without Cover_vision — still valid
        assert is_rvc_configured(enabled=False) is False
        return
    assert is_rvc_configured() is True


def test_rvc_enabled_without_cmd_falls_back_to_cover_or_skip(monkeypatch, tmp_path):
    monkeypatch.setenv("RVC_ENABLED", "1")
    monkeypatch.setenv("RVC_DISABLE_AUTO", "1")
    monkeypatch.delenv("RVC_INFER_CMD", raising=False)
    monkeypatch.delenv("RVC_MODEL_PATH", raising=False)
    inp = tmp_path / "in.wav"
    inp.write_bytes(b"RIFF")
    result = convert_vocals(inp, tmp_path / "out.wav", enabled=True)
    # With auto disabled and no template, should skip_no_infer_cmd
    assert result.status == "skipped_no_infer_cmd"
