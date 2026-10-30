import sys
import inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgical_vocal_variant import remix_surgical  # noqa: E402


def test_variant_module_exposes_vocal_only_workflow():
    assert callable(remix_surgical)
    params = inspect.signature(remix_surgical).parameters
    assert params["vocal_noise_profile"].default == "transparent"
    assert params["ratio_thresh"].default == 3.0
