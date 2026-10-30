"""Fold PEFT-style LoRA adapters into YuE2-3B's linear layers.

The downloaded YuE2 LoRAs (``yue2-steps-from-hell``, ``instrumental-cot``)
ship tensors named exactly like YuE2's own modules::

    model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_A / .lora_B     # AR/plan
    model.layers.{i}.mlp.{gate,up,down}_proj.lora_A / .lora_B
    model.layers.{i}.nar_self_attn.* / .nar_mlp.*                  # NAR/timbre
    vae2llm.lora_A / llm2vae.lora_A

YuE2's inference package (yue2-infer 0.1.3/0.1.5) has no LoRA hooks, but the
module names line up, so the delta can simply be folded into the base weight
before generation -- the same recipe the adapter authors ship in
``models/yue2/loras/instrumental-cot/scripts/ar_generate.py``::

    W += scale * alpha / rank * (B @ A)

One merge per process: cheap, keeps the CUDA-graph/torch-eager sampler
untouched, and leaves the vendor package unmodified.
"""
from __future__ import annotations

from pathlib import Path

import torch

_ADAPTER_KEYS = ("self_attn", "mlp", "nar_self_attn", "nar_mlp")


def _resolve(root: torch.nn.Module, dotted: str):
    obj = root
    for part in dotted.split("."):
        if part.isdigit() and isinstance(obj, (list, torch.nn.ModuleList)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def merge_lora(model: torch.nn.Module, path: str | Path, *, scale: float = 1.0,
               nar: bool = False) -> dict:
    """Merge one adapter file into ``model`` in place.

    Returns a small report: how many linears were merged, the effective scale
    (alpha / rank * ``scale``) and the adapter tensors that could not be
    resolved (a non-empty list means the file targets a different model).
    """
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    # AR tensors are named `model.layers...` (inside the backbone); the NAR
    # audio projections `vae2llm` / `llm2vae` sit on the top-level module.
    backbone = model.model if hasattr(model, "model") else model

    merged = 0
    missing: list[str] = []
    scales: set[float] = set()
    with torch.no_grad():
        for name in sorted(tensors):
            if not name.endswith(".lora_A"):
                continue
            stem = name[: -len(".lora_A")]
            b_name = stem + ".lora_B"
            if b_name not in tensors:
                missing.append(stem)
                continue
            if stem.startswith("model."):
                root, target = backbone, stem[len("model."):]
            elif stem.startswith("layers."):
                # instrumental-cot ships backbone keys without the `model.`
                # prefix; vae2llm/llm2vae stay on the top-level module.
                root, target = backbone, stem
            else:
                root, target = model, stem
            try:
                linear = _resolve(root, target)
            except AttributeError:
                missing.append(stem)
                continue
            if not isinstance(linear, torch.nn.Linear):
                missing.append(stem)
                continue
            a = tensors[name].to(linear.weight.device).float()
            b = tensors[b_name].to(linear.weight.device).float()
            rank = float(a.shape[0])
            alpha = rank
            # Adapters that ship an adapter_config.json carry the real alpha;
            # both downloaded sets use alpha == rank (effective 1.0), but read
            # it so a future adapter with alpha != rank is not silently wrong.
            cfg_path = Path(path).parent / "adapter_config.json"
            if cfg_path.exists():
                import json
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                rank = float(cfg.get("rank", rank))
                alpha = float(cfg.get("alpha", rank))
            eff = alpha / rank * scale if rank else scale
            delta = (b @ a) * eff
            linear.weight.add_(delta.to(linear.weight.dtype))
            merged += 1
            scales.add(round(eff, 4))
    return {"file": str(path), "merged": merged, "missing": missing,
            "scales": sorted(scales), "nar": nar}


def merge_pair(model: torch.nn.Module, ar: str | Path | None,
               nar: str | Path | None, *, ar_scale: float = 1.0,
               nar_scale: float = 1.0) -> list[dict]:
    reports: list[dict] = []
    if ar:
        reports.append(merge_lora(model, ar, scale=ar_scale, nar=False))
    if nar:
        reports.append(merge_lora(model, nar, scale=nar_scale, nar=True))
    return reports
