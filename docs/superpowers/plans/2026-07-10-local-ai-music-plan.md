# 本地 AI 音乐（编曲优先）实施计划书

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本地落地「编曲房（DeepSeek）+ 专业创作页 + ACE-Step 1.5 出歌」闭环，UI 信息架构对齐 ai6666 专业模式/编曲房。

**Architecture:** Next.js 前端 + FastAPI 网关；编排走 DeepSeek；出歌走本机 GPU 上的 ACE-Step Worker；任务异步、音频落盘、SQLite 索引。

**Tech Stack:** Next.js (TypeScript)、FastAPI、SQLite、DeepSeek API、ACE-Step 1.5、CUDA、Windows 本机优先。

**依据规格:** `docs/superpowers/specs/2026-07-10-local-ai-music-design.md`

---

## 文件结构（拟创建）

```
G:/AI_Music/
├── README.md
├── .env.example
├── .gitignore
├── docs/superpowers/...
├── apps/
│   └── web/                      # Next.js 前端
│       ├── package.json
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx          # 重定向到 /create
│       │   ├── create/page.tsx   # 专业创作
│       │   ├── studio/page.tsx   # 编曲房
│       │   └── library/page.tsx  # 我的
│       ├── components/
│       │   ├── Nav.tsx
│       │   ├── StructureTags.tsx
│       │   ├── StylePicker.tsx
│       │   ├── JobStatus.tsx
│       │   └── AudioPlayer.tsx
│       └── lib/api.ts
└── services/
    └── api/                      # FastAPI + Worker
        ├── pyproject.toml / requirements.txt
        ├── app/
        │   ├── main.py
        │   ├── config.py
        │   ├── db.py
        │   ├── models.py
        │   ├── schemas.py
        │   ├── routers/
        │   │   ├── arrange.py
        │   │   ├── generate.py
        │   │   ├── jobs.py
        │   │   └── library.py
        │   ├── services/
        │   │   ├── deepseek_arrange.py
        │   │   ├── acestep_runner.py
        │   │   └── queue.py
        │   └── prompts/
        │       └── arrange_system.txt
        └── tests/
            ├── test_arrange_schema.py
            └── test_jobs_api.py
```

---

## 阶段总览

| 阶段 | 目标 | 预估 |
|------|------|------|
| P0 | 推理通：歌词+风格 → 本地音频 | 1–2 天（含下模型） |
| P1 | 编曲房 + 专业创作 UI 打通 | 2–3 天 |
| P2 | 曲库、风格库、体验打磨 | 1–2 天 |
| P3 | 增强（描述模式/LoRA 等） | 按需 |

---

## P0：本地出歌最小闭环

### Task 1: 仓库与环境基线

**Files:**
- Create: `README.md`
- Create: `.env.example`
- Create: `services/api/requirements.txt`
- Modify: `.gitignore`（已有则核对）

- [ ] **Step 1:** 写 `README.md`：项目目标、硬件要求、快速启动三步（装依赖 / 下模型 / 启服务）
- [ ] **Step 2:** 写 `.env.example`：

```env
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
ACESTEP_MODEL_DIR=./models/ace-step-1.5
OUTPUT_DIR=./outputs
DATABASE_URL=sqlite:///./data/app.db
API_HOST=127.0.0.1
API_PORT=8000
```

- [ ] **Step 3:** 确认本机 `nvidia-smi` 可用，记录 GPU 型号与显存，写入 README「已验证环境」小节
- [ ] **Step 4:** Commit：`chore: bootstrap repo layout and env template`

### Task 2: FastAPI 骨架 + 健康检查

**Files:**
- Create: `services/api/app/main.py`
- Create: `services/api/app/config.py`
- Create: `services/api/requirements.txt`
- Test: `services/api/tests/test_health.py`

- [ ] **Step 1:** 写失败测试：`GET /health` 返回 `{"status":"ok"}`
- [ ] **Step 2:** 跑测试确认失败
- [ ] **Step 3:** 实现 FastAPI `main.py` + `config.py`（pydantic-settings）
- [ ] **Step 4:** 测试通过
- [ ] **Step 5:** Commit：`feat(api): add FastAPI health endpoint`

### Task 3: 任务表与 Job API

**Files:**
- Create: `services/api/app/db.py`
- Create: `services/api/app/models.py`
- Create: `services/api/app/schemas.py`
- Create: `services/api/app/routers/jobs.py`
- Create: `services/api/app/routers/generate.py`
- Test: `services/api/tests/test_jobs_api.py`

- [ ] **Step 1:** 定义 Job 字段：`id, title, lyrics, style_prompt, status, audio_path, error, created_at, updated_at`
- [ ] **Step 2:** 写测试：`POST /generate` → `queued`；`GET /jobs/{id}` 可读
- [ ] **Step 3:** 实现 SQLite + 路由（先不接真实模型，Worker 可写假 wav）
- [ ] **Step 4:** 测试通过
- [ ] **Step 5:** Commit：`feat(api): job create and status endpoints`

### Task 4: 接入 ACE-Step 1.5 推理

**Files:**
- Create: `services/api/app/services/acestep_runner.py`
- Create: `services/api/app/services/queue.py`
- Modify: `services/api/app/routers/generate.py`
- Modify: `README.md`（模型下载步骤）

- [ ] **Step 1:** 按官方仓库文档下载 ACE-Step 1.5 权重到 `models/ace-step-1.5`（目录 gitignore）
- [ ] **Step 2:** 实现 `acestep_runner.generate(lyrics, style_prompt, out_path, instrumental=False) -> Path`
- [ ] **Step 3:** Worker 从队列取 Job，调用 runner，更新 status / audio_path
- [ ] **Step 4:** 手工验收：用一段带 `[Verse]/[Chorus]` 的中文歌词生成并播放
- [ ] **Step 5:** Commit：`feat(api): integrate ACE-Step local generation worker`

**验收标准：** 不经过前端，仅用 curl/httpie 能拿到可播放音频。

---

## P1：编曲房 + 专业创作 UI

### Task 5: DeepSeek 编排服务

**Files:**
- Create: `services/api/app/prompts/arrange_system.txt`
- Create: `services/api/app/services/deepseek_arrange.py`
- Create: `services/api/app/routers/arrange.py`
- Test: `services/api/tests/test_arrange_schema.py`

- [ ] **Step 1:** 固定输出 JSON schema（见设计规格 §3.2），system prompt 要求只输出 JSON
- [ ] **Step 2:** 写测试：对 mock LLM 响应做 schema 校验（缺字段则失败）
- [ ] **Step 3:** 实现 `POST /arrange`：`{requirement, lyrics?, language}` → 编排结果
- [ ] **Step 4:** 校验失败时自动重试 1 次；仍失败返回 422 + 原始文本片段
- [ ] **Step 5:** Commit：`feat(api): DeepSeek arrangement endpoint`

### Task 6: Next.js 应用骨架与导航

**Files:**
- Create: `apps/web/*`（create-next-app）
- Create: `apps/web/components/Nav.tsx`
- Create: `apps/web/lib/api.ts`

- [ ] **Step 1:** 初始化 Next.js + TS，配置代理或 `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000`
- [ ] **Step 2:** 顶栏：生成音乐 / 编曲房 / 我的
- [ ] **Step 3:** 三页空壳可路由
- [ ] **Step 4:** Commit：`feat(web): scaffold Next.js app with nav`

### Task 7: 编曲房页面

**Files:**
- Create: `apps/web/app/studio/page.tsx`
- Create: `apps/web/components/...`（按需）

- [ ] **Step 1:** 布局对齐参考：要求框、可选歌词、语言、开始编排
- [ ] **Step 2:** 调用 `POST /arrange`，展示结构化歌词与风格标签
- [ ] **Step 3:** 「一键带去创作」跳转 `/create?draft=...` 或 POST draft
- [ ] **Step 4:** 手工验收：从编曲房到创作页字段预填正确
- [ ] **Step 5:** Commit：`feat(web): studio arrangement flow`

### Task 8: 专业创作页

**Files:**
- Create: `apps/web/app/create/page.tsx`
- Create: `apps/web/components/StructureTags.tsx`
- Create: `apps/web/components/StylePicker.tsx`
- Create: `apps/web/components/JobStatus.tsx`
- Create: `apps/web/components/AudioPlayer.tsx`

- [ ] **Step 1:** 表单：歌曲要求、歌词、结构标签插入、模型占位（本地 ACE-Step）、语言、纯音乐、风格维度、标题
- [ ] **Step 2:** 提交 `POST /generate`，轮询 `GET /jobs/{id}`
- [ ] **Step 3:** 成功后播放器 + 下载
- [ ] **Step 4:** 手工验收完整闭环
- [ ] **Step 5:** Commit：`feat(web): professional create page with generation`

---

## P2：曲库与体验

### Task 9: 本地曲库

**Files:**
- Create: `services/api/app/routers/library.py`
- Create: `apps/web/app/library/page.tsx`

- [ ] **Step 1:** `GET /library` 按时间倒序列出成功任务
- [ ] **Step 2:** 前端列表：标题、风格摘要、播放、下载、失败原因
- [ ] **Step 3:** Commit：`feat: local library page`

### Task 10: 风格标签库与最优推荐

**Files:**
- Create: `services/api/app/data/style_tags.json`
- Create: `services/api/app/routers/styles.py`
- Modify: `apps/web/components/StylePicker.tsx`
- Modify: `services/api/app/services/deepseek_arrange.py`（可选：推荐接口）

- [ ] **Step 1:** 整理流派/情绪/乐器/人声/速度中英文映射表
- [ ] **Step 2:** 「最优风格推荐」：基于歌词+要求调 DeepSeek 或规则拼接 `style_prompt`
- [ ] **Step 3:** Commit：`feat: style tag library and recommend`

### Task 11: 打磨与文档

**Files:**
- Modify: `README.md`
- Create: `docs/runbook.md`（可选）

- [ ] **Step 1:** 一键启动说明（两个终端：api + web）或 docker-compose（若做）
- [ ] **Step 2:** 常见问题：CUDA、显存、DeepSeek key、模型路径
- [ ] **Step 3:** Commit：`docs: usage and troubleshooting`

---

## P3：增强（规格确认后再拆细）

- 描述模式（自然语言一键出歌）
- ACE-Step 编辑能力（repaint / cover / vocal-to-BGM）
- 人声分轨
- LoRA 个人风格
- 可选云端模型适配器（非默认）

---

## 测试策略

| 类型 | 内容 |
|------|------|
| 单元 | 编排 JSON schema、Job 状态机 |
| 集成 | `/arrange` mock LLM；`/generate` mock runner |
| 手工 GPU | 每周至少 1 条真实中文歌词出歌回归 |
| UI | 编曲房 → 创作 → 播放 主路径 |

原则：无 GPU 的 CI 只跑 mock；真实推理在本机手工门禁。

---

## 实施顺序（严格）

1. Task 1 → 2 → 3 → 4（先有声音，再做漂亮 UI）  
2. Task 5 → 6 → 7 → 8（编曲与创作打通）  
3. Task 9 → 10 → 11（曲库与打磨）  
4. P3 按需  

**禁止：** 在 ACE-Step 本地推理未通前大面积做视觉像素级复刻。

---

## 启动检查清单（实施前）

- [ ] 本机 NVIDIA 驱动 + CUDA 可用  
- [ ] 显存 ≥ 8GB（理想 12GB+）  
- [ ] 已准备 DeepSeek API Key  
- [ ] 磁盘预留 ≥ 30GB（模型 + 音频）  
- [ ] 用户已确认设计规格 `docs/superpowers/specs/2026-07-10-local-ai-music-design.md`

---

## 参考链接

- 产品参考：https://ai6666.com/music/ 、https://ai6666.com/music/studio/  
- ACE-Step 1.5：https://ace-step.github.io/ace-step-v1.5.github.io/  
- 设计规格：`docs/superpowers/specs/2026-07-10-local-ai-music-design.md`
