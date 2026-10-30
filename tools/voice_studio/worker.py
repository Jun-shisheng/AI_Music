"""Backend process. Run only with an explicit workflow execution request.

Uses official SoulX model APIs and custom prompt audio. Never substitutes bundled demo voices.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["f0", "svc", "svs", "probe"])
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    import numpy as np
    if args.task == "probe":
        import importlib
        report = {}
        for name in spec.get("modules",["torch", "torchaudio", "transformers", "omegaconf", "accelerate", "librosa", "soundfile"]):
            try:
                module = importlib.import_module(name)
                report[name] = getattr(module, "__version__", "available")
            except Exception as exc:
                report[name] = {"error": str(exc)}
        print(json.dumps(report))
        return
    import soundfile as sf
    if args.task == "f0":
        method = spec.get("method", "dio")
        if method == "pyin":
            # pyin at a 480-sample hop on 24 kHz audio is exactly the model's
            # 50 Hz F0 grid. Harvest is deliberately not used: in this build it
            # reports zero voiced frames on clean 220/440 Hz tones.
            import librosa
            for item in spec["files"]:
                wav, rate = sf.read(item["audio"], dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                f0, voiced, _ = librosa.pyin(
                    wav, fmin=spec.get("f0_floor", 65), fmax=spec.get("f0_ceil", 1600),
                    sr=rate, frame_length=2048, hop_length=480)
                f0 = np.where(np.asarray(voiced, dtype=bool), f0, 0.0)
                for index in np.where(f0 > 0)[0]:
                    neighbours = [f0[j] for j in range(max(0, index - 2), min(len(f0), index + 3))
                                  if j != index and f0[j] > 0]
                    if len(neighbours) >= 3:
                        centre = float(np.median(neighbours))
                        if abs(12.0 * np.log2(f0[index] / (centre + 1e-9))) > 4.5:
                            f0[index] = 0.0
                frames = int(np.ceil(len(wav) / rate * 50))
                f0 = np.pad(f0, (0, max(0, frames-len(f0))))[:frames]
                np.save(item["output"], f0.astype("float32"))
            return
        import pyworld
        for item in spec["files"]:
            wav, rate = sf.read(item["audio"], dtype="float64")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            f0, times = pyworld.dio(wav, rate, f0_floor=55, f0_ceil=1100, frame_period=20.)
            f0 = pyworld.stonemask(wav, f0, times, rate)
            frames = int(np.ceil(len(wav) / rate * 50))
            f0 = np.pad(f0, (0, max(0, frames-len(f0))))[:frames]
            np.save(item["output"], f0.astype("float32"))
        return

    os.environ.setdefault("HF_HOME", str(Path(spec["models"]) / "hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Upstream currently hardcodes the Whisper repo name; resolve to the prepared local snapshot.
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel
    for cls in (WhisperFeatureExtractor, WhisperModel):
        original = cls.from_pretrained
        def local_loader(name, *pos, _original=original, **kwargs):
            if name == "openai/whisper-base":
                name = spec["whisper"]
                kwargs["local_files_only"] = True
            return _original(name, *pos, **kwargs)
        cls.from_pretrained = local_loader
    sys.path.insert(0, spec["soulx_root"])
    from soulxsinger.utils.file_utils import load_config
    config = load_config(spec["config"])
    config.infer.n_steps = spec["steps"]
    config.infer.cfg = spec["cfg"]
    torch.manual_seed(spec["seed"])
    if spec["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this backend interpreter cannot access CUDA")
    if args.task == "svc":
        from cli.inference_svc import build_model
        from soulxsinger.utils.audio_utils import load_wav
        model = build_model(spec["model"], config, spec["device"], spec["fp16"])
        prompt_cache = {}

        def prompt_for(item):
            """Load the register prompt selected for this chunk, once per prompt."""
            if item.get("prompt_wav") and item.get("prompt_f0"):
                wav_path, f0_path = item["prompt_wav"], item["prompt_f0"]
                cache_key = f"{wav_path}|{f0_path}"
            else:
                name = item.get("prompt_register") or item.get("prompt")
                entry = next((p for p in spec.get("prompts", [])
                              if p.get("register") == name or p.get("key") == name), None)
                if entry:
                    wav_path, f0_path = entry["wav"], entry["f0"]
                    cache_key = f"{entry.get('register', name)}|{wav_path}|{f0_path}"
                else:
                    wav_path, f0_path = spec["prompt_wav"], spec["prompt_f0"]
                    cache_key = f"default|{wav_path}|{f0_path}"
            if cache_key not in prompt_cache:
                prompt_cache[cache_key] = (
                    load_wav(wav_path, 24000).to(spec["device"]),
                    torch.from_numpy(np.load(f0_path)).unsqueeze(0).to(spec["device"]),
                )
            return prompt_cache[cache_key]

        for item in spec["chunks"]:
            output = Path(item["output"])
            done = output.with_suffix(".done.json")
            if output.exists() and done.exists():
                state = json.loads(done.read_text())
                import hashlib
                if state.get("key") == item["key"] and state.get("sha256") == hashlib.sha256(output.read_bytes()).hexdigest():
                    continue
            target = load_wav(item["audio"], 24000).to(spec["device"])
            target_f0 = torch.from_numpy(np.load(item["f0"])).unsqueeze(0).to(spec["device"])
            prompt, prompt_f0 = prompt_for(item)
            with torch.inference_mode():
                generated, shift = model.infer(prompt, target, prompt_f0, target_f0,
                    auto_shift=False, pitch_shift=spec["pitch_shift"], n_steps=spec["steps"],
                    cfg=spec["cfg"], use_fp16=spec["fp16"])
            data = generated.squeeze().float().cpu().numpy()
            if len(data) != target.shape[-1] or not np.isfinite(data).all():
                raise RuntimeError(f"Invalid waveform for {output}")
            sf.write(str(output), data, 24000, subtype="PCM_16")
            import hashlib
            done.write_text(json.dumps(dict(key=item["key"], pitch_shift=int(shift),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())), encoding="utf-8")
            print(f"Converted {item['audio']} -> {output}", flush=True)
    else:
        from cli.inference import build_model, process
        from types import SimpleNamespace
        model = build_model(spec["model"], config, spec["device"], spec["fp16"])
        process(SimpleNamespace(device=spec["device"], prompt_wav_path=spec["prompt_wav"],
            prompt_metadata_path=spec["prompt_metadata"], target_metadata_path=spec["target_metadata"],
            phoneset_path=str(Path(spec["soulx_root"])/"soulxsinger/utils/phoneme/phone_set.json"),
            save_dir=spec["save_dir"], auto_shift=False, pitch_shift=spec["pitch_shift"],
            use_fp16=spec["fp16"], control=spec["control"]), config, model)


if __name__ == "__main__":
    main()
