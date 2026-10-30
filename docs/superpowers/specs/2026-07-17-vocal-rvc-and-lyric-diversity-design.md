# 修音增强 + 编曲变化优化 — 设计规格

> 日期：2026-07-17  
> 状态：已按用户确认开干（修音选 C：先 RVC 后 LoRA；编曲反重复默认 A+B+打分）  
> 背景：本地创作 / 打分 / Demucs+ffmpeg 修音已通；听感仍偏 AI，歌词与编曲变化不足

## 1. 目标

1. **人声更贴真人**：在现有 stem 流水线中接入可选的 UP 主音色转换（RVC），减少杂音断裂感；错音节仍走重生成。  
2. **降低歌词疲劳重复**：编排后硬性多样性检查 + 不达标自动改写；打分惩罚「无推进复读」。  
3. **为后续 ACE LoRA 留接口**：先攒高分+过耳素材，不阻塞本阶段上线。

## 2. 非目标（本阶段不做）

- 立刻训练并交付可用的 UP 主 LoRA / RVC 权重（需用户提供合法素材后另开任务）  
- ASR 对齐级人声 QA  
- 完整 DAW / 多轨编曲引擎  
- 替换 ACE-Step 骨干模型

## 3. 方案摘要

### 3.1 修音路径（已确认）

```
review WAV
  → Demucs 分轨
  → ffmpeg 轻量清理（transparent/gentle）
  → [可选] RVC 换声（配置模型后启用）
  → remix + loudnorm
  → top_songs
```

| 阶段 | 内容 |
|------|------|
| **P1（本迭代）** | RVC 适配层 + stem_pipeline 开关；无模型时优雅跳过 |
| **P2** | 用户提供 UP 主人声素材 → 训练 RVC → 默认开启换声 |
| **P3** | 高分曲库训 ACE-Step LoRA，从生成端减 AI 杂音 |

### 3.2 编曲变化 / 反重复（默认全开）

| 关卡 | 位置 | 规则要点 |
|------|------|----------|
| Prompt 强化 | `arrange_system.txt` + `_build_user_prompt` | 主歌禁止复读；副歌允许 hook 但需情绪递进；桥段必须新信息 |
| 硬校验 | `lyric_diversity.py` | 全曲 unique_ratio、主歌间相似、连续重复行、n-gram 疲劳 |
| 自动改写 | `deepseek_arrange.arrange` | 校验失败则带问题清单二次请求 DeepSeek |
| 打分 | `curation.score_track` | 「疲劳重复」扣 narrative/hook；有意的副歌 hook 重复仍可加分 |

**允许的重复：** 副歌核心 hook 行（短、可记）、标题回扣。  
**禁止的重复：** 主歌大段改写式复读、连续 3+ 行相同、全曲 unique_ratio 过低且无结构理由。

## 4. 架构落点

| 模块 | 路径 |
|------|------|
| RVC 适配 | `tools/vocal_repair/rvc_convert.py` |
| Stem 串联 | `tools/vocal_repair/stem_pipeline.py` |
| 歌词多样性 | `services/api/app/services/lyric_diversity.py` |
| 编排接入 | `services/api/app/services/deepseek_arrange.py` |
| 打分 | `tools/social_media/curation.py` |
| 工作流透传 | `tools/social_media/top_song_workflow.py`（可选 `--rvc-*`） |

## 5. RVC 配置约定

环境变量 / CLI：

- `RVC_ENABLED=1`  
- `RVC_MODEL_PATH` — `.pth` 或项目约定目录  
- `RVC_INDEX_PATH` — 可选 faiss index  
- `RVC_INFER_CMD` — 可选自定义推理命令模板（含 `{input}` `{output}` `{model}`）

无配置时：`metadata.steps.rvc = {status: "skipped_not_configured"}`，流水线不中断。

## 6. 成功标准

1. 未配置 RVC 时，现有 stem 流程行为与现在一致。  
2. 配置假/真推理命令后，`metadata.json` 能记录 `converted_vocals` 并用于 remix。  
3. 故意高度重复的编排 JSON 在 `lyric_diversity` 下失败；经改写或拒收后 unique_ratio 达标。  
4. 打分对「主歌复读型」歌词总分低于改进前同结构对照。  

## 8. Cover_vision 默认模型（已接通）

本机若存在 `G:\Cover_vision`，自动使用：

- 推理：`Cover_vision\.venv\python.exe` + `tools\rvc_infer.py`
- 权重：`rvc-webui\assets\weights\vp_v2_48k.pth`（导出权重；勿用 `*_e20_s*.pth`）
- 索引：`rvc-webui\logs\vp_v2_48k\added_IVF256_Flat.index`

冒烟（2026-07-17）：25s clip → Demucs → RVC `succeeded` → remix 使用 `rvc_vocals.wav`。

## 7. 风险

| 风险 | 对策 |
|------|------|
| Cover_vision 路径搬迁 | 设 `COVER_VISION_ROOT` 或 `RVC_INFER_CMD` |
| 误用训练快照 `*_e20_s*.pth` | 默认只用导出权重 `vp_v2_48k.pth` |
| RVC 仍修不好错词 | 维持 `needs_regen`；换声不做歌词修复 |
| 反重复过严杀 hook | 副歌短 hook 白名单；打分区分「有用重复」 |
| DeepSeek 二次改写漂移 | schema 再校验；最多 2 次改写重试 |
