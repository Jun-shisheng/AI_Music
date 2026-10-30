# 修音 RVC + 歌词反重复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 stem 修音链路接入可选 RVC 换声；在编排与打分链路加入硬性歌词多样性关卡，降低疲劳重复。

**Architecture:** RVC 作为 `stem_pipeline` 中 Demucs+ffmpeg 之后的可选步骤；歌词多样性为纯函数模块，挂在 DeepSeek `arrange()` 校验之后并可触发一次改写；`curation.score_track` 同步惩罚疲劳重复。

**Tech Stack:** Python、ffmpeg、Demucs（已有）、RVC 外部推理（可选）、DeepSeek、pytest

**Spec:** `docs/superpowers/specs/2026-07-17-vocal-rvc-and-lyric-diversity-design.md`

---

## 文件结构

- Create: `tools/vocal_repair/rvc_convert.py`
- Modify: `tools/vocal_repair/stem_pipeline.py`
- Modify: `tools/vocal_repair/README.md`
- Modify: `tools/social_media/top_song_workflow.py`
- Create: `services/api/app/services/lyric_diversity.py`
- Modify: `services/api/app/services/deepseek_arrange.py`
- Modify: `services/api/app/prompts/arrange_system.txt`
- Modify: `tools/social_media/curation.py`
- Create: `services/api/tests/test_lyric_diversity.py`
- Create: `tools/vocal_repair/tests/test_rvc_convert.py`（或 api 侧 mock 测 skip 逻辑）
- Modify: `.env.example`（若存在）

---

### Task 1: 歌词多样性纯函数 + 测试

**Files:**
- Create: `services/api/app/services/lyric_diversity.py`
- Create: `services/api/tests/test_lyric_diversity.py`

- [ ] **Step 1:** 写失败测试：高重复歌词 `passes=False`；合理副歌 hook 重复 `passes=True`
- [ ] **Step 2:** 实现 `analyze_lyrics` / `assert_lyric_diversity`（unique_ratio、连续重复、主歌块相似、n-gram）
- [ ] **Step 3:** 测试通过
- [ ] **Step 4:** Commit（若用户要求）

### Task 2: 编排接入多样性关卡

**Files:**
- Modify: `services/api/app/services/deepseek_arrange.py`
- Modify: `services/api/app/prompts/arrange_system.txt`
- Modify: `services/api/tests/test_arrange_schema.py`

- [ ] **Step 1:** system/user prompt 写入反疲劳规则
- [ ] **Step 2:** `arrange()` 在 `_validate_arrangement` 后跑多样性；失败则带 issues 改写一次
- [ ] **Step 3:** 测试 mock 场景或单元测改写路径
- [ ] **Step 4:** Commit

### Task 3: 打分惩罚疲劳重复

**Files:**
- Modify: `tools/social_media/curation.py`

- [ ] **Step 1:** 引入与 `lyric_diversity` 一致的启发式（或轻量内联，避免跨包硬依赖时可复制核心指标）
- [ ] **Step 2:** unique_ratio 过低 / 主歌复读 → narrative 扣分并写 reasons
- [ ] **Step 3:** 保留「1–8 行有用副歌重复」的 hook 加分

### Task 4: RVC 适配层

**Files:**
- Create: `tools/vocal_repair/rvc_convert.py`
- Create: `tools/vocal_repair/tests/test_rvc_skip.py`（可选）

- [ ] **Step 1:** `is_rvc_configured()` / `convert_vocals(input, output, ...)`  
  - 未配置 → 返回 skipped  
  - 配置 `RVC_INFER_CMD` → subprocess 模板替换  
- [ ] **Step 2:** 单元测：未配置时 status=skipped_not_configured

### Task 5: stem_pipeline + top_song_workflow 串联

**Files:**
- Modify: `tools/vocal_repair/stem_pipeline.py`
- Modify: `tools/social_media/top_song_workflow.py`
- Modify: `tools/vocal_repair/README.md`
- Modify: `.env.example`

- [ ] **Step 1:** `process_file` 在 repair 后调用 RVC；成功则 remix 用 converted 人声
- [ ] **Step 2:** CLI：`--rvc` / `--rvc-model` 等
- [ ] **Step 3:** workflow 透传环境或参数
- [ ] **Step 4:** README 写清：素材→训练→配置→启用

### Task 6: 冒烟

- [ ] 跑 `pytest services/api/tests/test_lyric_diversity.py`
- [ ] 跑既有 arrange/jobs 测试不回归
- [ ] 无 RVC 时 stem_pipeline `--limit 0` 或 dry 逻辑不炸

---

## 实施顺序

Task 1 → 2 → 3 → 4 → 5 → 6
