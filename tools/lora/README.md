# ACE-Step LoRA (AI_Music)

> **⚠️ 已废弃（2026-10-16 归档确认）**：LoRA 训练从未真正生效——`findings.md` 记录
> "LoRA load returned 400 every attempt"，服务一直用基座模型生成。数据集
> （`data/lora_datasets`、`data/lora_tensors`，共约 218 MB）仍在磁盘上，未清理。

从高分曲库导出数据集，并封装 ACE-Step 官方 `train.py fixed` 训练流程。

## 前置

- 官方建议显存 ≥16GB；**RTX 4060 8GB** 用 `--minimal-vram`（45s 裁剪 + rank 16 + offload-encoder）
- ACE-Step `.venv` 已含 `peft`；可选 `bitsandbytes`
- 已有 `outputs/top_songs` 或 DB 高分曲
- 训练前尽量关掉 ACE API / Gradio，腾出显存

## 1. 导出数据集

```powershell
cd G:\AI_Music
python tools/lora/export_dataset.py --min-score 85
python tools/lora/export_dataset.py --from-db --min-score 85 --limit 30
```

输出：`data/lora_datasets/from_top_songs/dataset.json` + `audio/`

## 2. 预览命令

```powershell
python tools/lora/train_lora.py --dry-run --minimal-vram
```

## 3. 预处理 + 训练（8GB）

```powershell
python tools/lora/train_lora.py --minimal-vram --preprocess-only
python tools/lora/train_lora.py --train-only --minimal-vram --epochs 15
```

产出：`vendor/ACE-Step-1.5/lora_workspace/output/from_top_songs/final`

默认模型：`vp_v2_48k.pth`（导出权重）。勿用 `*_e20_s*.pth` 训练快照。

## 4. 推理加载

`.env`：

```env
ACESTEP_BACKEND=api
ACESTEP_LORA_PATH=G:/AI_Music/vendor/ACE-Step-1.5/lora_workspace/output/from_top_songs/final
ACESTEP_LORA_SCALE=0.8
ACESTEP_LORA_ADAPTER_NAME=ai_music
```

重启 API 后首次生成会 `/v1/lora/load`。

## 与 RVC

- **RVC**：成品换声（Cover_vision）
- **LoRA**：生成侧风格  
可同时：LoRA 出歌 → top 工作流 RVC。
