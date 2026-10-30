import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yue2_lora import merge_lora, merge_pair  # noqa: E402


class _Attn(torch.nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.o_proj = torch.nn.Linear(dim, dim, bias=False)


class _Mlp(torch.nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.gate_proj = torch.nn.Linear(dim, dim, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer()])


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()


def _adapter(tmp_path, prefix="model.layers.0", a_scale=1.0, b_scale=1.0, rank=2):
    torch.manual_seed(0)
    a = torch.randn(rank, 4) * a_scale
    b = torch.randn(4, rank) * b_scale
    path = tmp_path / "lora.safetensors"
    save_file({f"{prefix}.self_attn.q_proj.lora_A": a,
               f"{prefix}.self_attn.q_proj.lora_B": b}, str(path))
    return path, a, b


def test_merge_adds_b_times_a(tmp_path):
    model = _FakeModel()
    path, a, b = _adapter(tmp_path)
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()

    report = merge_lora(model, path, scale=0.5)

    after = model.model.layers[0].self_attn.q_proj.weight.detach()
    expected = before + 0.5 * (b @ a)
    assert torch.allclose(after, expected, atol=1e-6)
    assert report["merged"] == 1
    assert report["missing"] == []
    assert report["scales"] == [0.5]


def test_unresolved_keys_are_reported_not_fatal(tmp_path, capsys):
    model = _FakeModel()
    path = tmp_path / "lora.safetensors"
    save_file({"layers.0.self_attn.k_proj.lora_A": torch.randn(2, 4),
               "layers.0.self_attn.k_proj.lora_B": torch.randn(4, 2)}, str(path))

    report = merge_lora(model, path)

    assert report["merged"] == 0
    assert report["missing"] == ["layers.0.self_attn.k_proj"]


def test_backbone_keys_without_model_prefix_resolve(tmp_path):
    # instrumental-cot ships `layers.{i}...`, the TSFH set ships
    # `model.layers.{i}...`; both must reach the same backbone modules.
    model = _FakeModel()
    path, a, b = _adapter(tmp_path, prefix="layers.0")
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()

    report = merge_lora(model, path)

    after = model.model.layers[0].self_attn.q_proj.weight.detach()
    assert torch.allclose(after, before + (b @ a), atol=1e-6)
    assert report["merged"] == 1


def test_merge_pair_handles_ar_only(tmp_path):
    model = _FakeModel()
    path, a, b = _adapter(tmp_path)

    reports = merge_pair(model, path, None, ar_scale=1.0)

    assert len(reports) == 1
    assert reports[0]["merged"] == 1
    assert reports[0]["nar"] is False


def test_adapter_config_alpha_is_honoured(tmp_path):
    import json

    model = _FakeModel()
    path, a, b = _adapter(tmp_path, rank=2)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"rank": 2, "alpha": 4.0}), encoding="utf-8")
    before = model.model.layers[0].self_attn.q_proj.weight.detach().clone()

    merge_lora(model, path, scale=0.5)

    after = model.model.layers[0].self_attn.q_proj.weight.detach()
    # alpha / rank = 2, times caller scale 0.5 -> 1.0
    assert torch.allclose(after, before + (b @ a), atol=1e-6)
