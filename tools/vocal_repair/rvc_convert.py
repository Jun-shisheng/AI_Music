"""Optional RVC / timbre conversion for isolated vocal stems.

Auto-detects G:\\Cover_vision vp_v2_48k assets when present so AI_Music
can convert without manually setting RVC_INFER_CMD.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


COVER_VISION_ROOT = Path(os.environ.get("COVER_VISION_ROOT", r"G:\Cover_vision"))
COVER_VISION_PYTHON = COVER_VISION_ROOT / ".venv" / "python.exe"
COVER_VISION_INFER = COVER_VISION_ROOT / "tools" / "rvc_infer.py"
COVER_VISION_MODEL_FINAL = COVER_VISION_ROOT / "rvc-webui" / "assets" / "weights" / "vp_v2_48k.pth"
COVER_VISION_MODEL_E20 = (
    COVER_VISION_ROOT / "rvc-webui" / "assets" / "weights" / "vp_v2_48k_e20_s17800.pth"
)
COVER_VISION_INDEX = (
    COVER_VISION_ROOT / "rvc-webui" / "logs" / "vp_v2_48k" / "added_IVF256_Flat.index"
)


@dataclass
class RvcResult:
    status: str
    input: str
    output: str | None = None
    command: str | None = None
    error: str | None = None
    preset: str | None = None


@dataclass
class CoverVisionPreset:
    python: Path
    infer_script: Path
    model_path: Path
    index_path: Path

    @property
    def ready(self) -> bool:
        return (
            self.python.is_file()
            and self.infer_script.is_file()
            and self.model_path.is_file()
            and self.index_path.is_file()
        )


def discover_cover_vision() -> CoverVisionPreset | None:
    # Prefer the exported inference weight registered in rvc_library/models.json.
    # Epoch checkpoints like *_e20_s*.pth are training snapshots and often lack "config".
    if COVER_VISION_MODEL_FINAL.is_file():
        model = COVER_VISION_MODEL_FINAL
    elif COVER_VISION_MODEL_E20.is_file():
        model = COVER_VISION_MODEL_E20
    else:
        return None
    preset = CoverVisionPreset(
        python=COVER_VISION_PYTHON,
        infer_script=COVER_VISION_INFER,
        model_path=model,
        index_path=COVER_VISION_INDEX,
    )
    return preset if preset.ready else None


def is_rvc_configured(
    enabled: bool | None = None,
    model_path: str | Path | None = None,
    infer_cmd: str | None = None,
) -> bool:
    if enabled is False:
        return False
    if enabled is True:
        return True
    env_enabled = os.environ.get("RVC_ENABLED", "").strip().lower() in {"1", "true", "yes"}
    if env_enabled:
        return True
    if infer_cmd or os.environ.get("RVC_INFER_CMD"):
        return True
    if model_path or os.environ.get("RVC_MODEL_PATH"):
        return True
    # Auto-enable when Cover_vision residual model is on disk
    if os.environ.get("RVC_DISABLE_AUTO", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return discover_cover_vision() is not None


def _cover_vision_cmd(
    input_wav: Path,
    output_wav: Path,
    model_path: str | Path | None,
    index_path: str | Path | None,
) -> tuple[list[str], str] | None:
    if os.environ.get("RVC_DISABLE_AUTO", "").strip().lower() in {"1", "true", "yes"}:
        return None
    preset = discover_cover_vision()
    if preset is None:
        return None
    model = Path(model_path) if model_path else preset.model_path
    index = Path(index_path) if index_path else preset.index_path
    if not model.is_file():
        model = preset.model_path
    if not index.is_file():
        index = preset.index_path
    cmd = [
        str(preset.python),
        str(preset.infer_script),
        "--model_path",
        str(model),
        "--index_path",
        str(index),
        "--input",
        str(input_wav),
        "--output",
        str(output_wav),
        "--f0_method",
        "rmvpe",
        "--f0_up_key",
        "0",
        "--index_rate",
        "0.75",
        "--device",
        os.environ.get("RVC_DEVICE", "cuda"),
    ]
    return cmd, f"cover_vision:{model.name}"


def _resolve_infer_cmd(
    input_wav: Path,
    output_wav: Path,
    model_path: str | Path | None,
    index_path: str | Path | None,
    infer_cmd: str | None,
) -> tuple[list[str], str | None] | None:
    template = infer_cmd or os.environ.get("RVC_INFER_CMD", "").strip()
    model = str(model_path or os.environ.get("RVC_MODEL_PATH", "")).strip()
    index = str(index_path or os.environ.get("RVC_INDEX_PATH", "")).strip()

    if template:
        filled = (
            template.replace("{input}", str(input_wav))
            .replace("{output}", str(output_wav))
            .replace("{model}", model)
            .replace("{index}", index)
        )
        return shlex.split(filled, posix=os.name != "nt"), "custom_template"

    auto = _cover_vision_cmd(input_wav, output_wav, model_path or model or None, index_path or index or None)
    if auto:
        return auto

    return None


def convert_vocals(
    input_wav: Path,
    output_wav: Path,
    *,
    enabled: bool | None = None,
    model_path: str | Path | None = None,
    index_path: str | Path | None = None,
    infer_cmd: str | None = None,
) -> RvcResult:
    input_wav = Path(input_wav).resolve()
    output_wav = Path(output_wav).resolve()

    if not is_rvc_configured(enabled=enabled, model_path=model_path, infer_cmd=infer_cmd):
        return RvcResult(
            status="skipped_not_configured",
            input=str(input_wav),
            output=None,
        )

    resolved = _resolve_infer_cmd(input_wav, output_wav, model_path, index_path, infer_cmd)
    if not resolved:
        return RvcResult(
            status="skipped_no_infer_cmd",
            input=str(input_wav),
            output=None,
            error=(
                "RVC enabled but no infer command and Cover_vision defaults missing. "
                "Set RVC_INFER_CMD or install G:\\Cover_vision RVC assets."
            ),
        )

    cmd, preset_name = resolved
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        return RvcResult(
            status="failed",
            input=str(input_wav),
            output=None,
            command=" ".join(cmd),
            error=str(exc),
            preset=preset_name,
        )

    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
        return RvcResult(
            status="failed",
            input=str(input_wav),
            output=None,
            command=" ".join(cmd),
            error=err[-2000:],
            preset=preset_name,
        )

    if not output_wav.exists():
        return RvcResult(
            status="failed",
            input=str(input_wav),
            output=None,
            command=" ".join(cmd),
            error=f"RVC command succeeded but output missing: {output_wav}",
            preset=preset_name,
        )

    return RvcResult(
        status="succeeded",
        input=str(input_wav),
        output=str(output_wav),
        command=" ".join(cmd),
        preset=preset_name,
    )
