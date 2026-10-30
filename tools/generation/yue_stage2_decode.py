from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from einops import rearrange
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YUE_INFERENCE = PROJECT_ROOT / "vendor" / "YuE" / "inference"
XCODEC = YUE_INFERENCE / "xcodec_mini_infer"
STAGE2 = PROJECT_ROOT / "models" / "yue" / "YuE-s2-1B-general"
sys.path.insert(0, str(YUE_INFERENCE))
sys.path.insert(0, str(XCODEC))
sys.path.insert(0, str(XCODEC / "descriptaudiocodec"))

from codecmanipulator import CodecManipulator  # noqa: E402
from mmtokenizer import _MMSentencePieceTokenizer  # noqa: E402
from models.soundstream_hubert_new import SoundStream  # noqa: E402,F401


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id: int, end_id: int):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def stage2_generate(model, prompt, tokenizer, codectool, device):
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
        codec_ids,
        global_offset=codectool.global_offset,
        codebook_size=codectool.codebook_size,
        num_codebooks=codectool.num_codebooks,
    ).astype(np.int32)
    prompt_ids = np.concatenate([
        np.array([tokenizer.soa, tokenizer.stage_1]),
        codec_ids.flatten(),
        np.array([tokenizer.stage_2]),
    ]).astype(np.int32)[None, ...]
    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    prompt_len = prompt_ids.shape[-1]
    block_list = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 46358),
        BlockTokenRangeProcessor(53526, tokenizer.vocab_size),
    ])
    for frame in range(codec_ids.shape[1]):
        prompt_ids = torch.cat([prompt_ids, codec_ids[:, frame:frame + 1]], dim=1)
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=tokenizer.eoa,
                pad_token_id=tokenizer.eoa,
                logits_processor=block_list,
            )
        prompt_ids = generated
    return prompt_ids[0].cpu().numpy()[prompt_len:]


def fix_codes(output: np.ndarray) -> np.ndarray:
    fixed = copy.deepcopy(output)
    for row, line in enumerate(output):
        for col, element in enumerate(line):
            if element < 0 or element > 1023:
                counts = Counter(line)
                fixed[row, col] = counts.most_common(1)[0][0]
    return fixed


def save_audio(waveform: torch.Tensor, path: Path, sample_rate: int) -> None:
    waveform = waveform.detach().cpu().squeeze(0).clamp(-0.99, 0.99)
    torchaudio.save(str(path), waveform, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode cached YuE Stage 1 stems, including sub-6s smoke tests.")
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    args.stage1_dir = args.stage1_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(YUE_INFERENCE)
    stems = sorted(list(args.stage1_dir.glob("*_itrack.npy")) + list(args.stage1_dir.glob("*_vtrack.npy")))
    if not stems:
        raise FileNotFoundError(f"No Stage 1 stems found in {args.stage1_dir}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = _MMSentencePieceTokenizer(str(YUE_INFERENCE / "mm_tokenizer_v0.2_hf" / "tokenizer.model"))
    codectool = CodecManipulator("xcodec", 0, 8)
    model = AutoModelForCausalLM.from_pretrained(
        str(STAGE2), torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device).eval()
    config = OmegaConf.load(XCODEC / "final_ckpt" / "config.yaml")
    codec_model = eval(config.generator.name)(**config.generator.config).to(device)
    checkpoint = torch.load(XCODEC / "final_ckpt" / "ckpt_00360000.pth", map_location="cpu", weights_only=False)
    codec_model.load_state_dict(checkpoint["codec_model"])
    codec_model.eval()
    decoded = {}
    for stem in stems:
        prompt = np.load(stem).astype(np.int32)
        if args.max_frames:
            prompt = prompt[:, :args.max_frames]
        output = fix_codes(CodecManipulator("xcodec", 0, 8).ids2npy(
            stage2_generate(model, prompt, tokenizer, CodecManipulator("xcodec", 0, 1), device)
        ))
        output_path = args.output_dir / f"{stem.stem}.npy"
        np.save(output_path, output)
        codec_result = torch.as_tensor(output.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device)
        with torch.no_grad():
            waveform = codec_model.decode(codec_result)
        wav_path = args.output_dir / f"{stem.stem}.wav"
        save_audio(waveform, wav_path, 16000)
        decoded[stem.stem] = torchaudio.load(str(wav_path))[0].squeeze(0).numpy()
    vocal = next((value for key, value in decoded.items() if "_vtrack" in key), None)
    inst = next((value for key, value in decoded.items() if "_itrack" in key), None)
    if vocal is not None and inst is not None:
        length = min(len(vocal), len(inst))
        mix = (vocal[:length] + inst[:length]).clip(-0.99, 0.99)
        sf.write(args.output_dir / "yue_smoke_mix.wav", mix, 16000)
    print(f"Decoded {len(decoded)} stems to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
