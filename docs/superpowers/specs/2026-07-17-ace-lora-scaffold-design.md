# ACE-Step LoRA 训练脚手架 — 设计规格

> 日期：2026-07-17  
> 状态：实施中

## 目标

用本机高分过耳曲（优先 `outputs/top_songs`，回退 SQLite `jobs`）导出 ACE-Step 训练数据集，并提供 preprocess / train / 推理加载脚手架，从生成端减少 AI 杂音、贴近目标音色。

## 范围

| 做 | 不做（本迭代） |
|----|----------------|
| 导出 `dataset.json` + 并列 `.lyrics.txt` / `.json` | 自动跑满 GPU 训练到收敛（需本机 16GB+ 与 peft） |
| `tools/lora/train_lora.py` 封装 preprocess+train | 多卡 / Muse 大规模 SFT |
| `ACESTEP_LORA_PATH` → API `/v1/lora/load` | 替换 Cover_vision RVC（两者可并存） |

## 数据映射

| ACE 字段 | 来源 |
|----------|------|
| `audio_path` | `top_songs/*.wav`（修音后）或 `jobs.audio_path` |
| `lyrics` | top `.txt` 去头，或 `jobs.lyrics` |
| `caption` | `jobs.style_prompt` / sidecar |
| `language` | `zh` |

## 成功标准

1. `python tools/lora/export_dataset.py` 能从现有 5 首 top_songs 写出合法 dataset  
2. `train_lora.py --dry-run` 打印可复制的 `train.py fixed` 命令  
3. 配置 `ACESTEP_LORA_PATH` 后，API 生成前会尝试 load LoRA（路径缺失则跳过并打日志）
