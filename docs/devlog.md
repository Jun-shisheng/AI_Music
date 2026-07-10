# 开发日志

## 2026-07-10 — 项目初始化与 P0 后端闭环

### 环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM) |
| Python (API) | 3.13.13 |
| Python (ACE-Step) | 3.11.15 (conda env: `acestep`) |
| CUDA | 待安装 CUDA 版 PyTorch |
| OS | Windows 11 Home China 10.0.22631 |
| 项目路径 | `G:/AI_Music` |

### 完成事项

#### 仓库初始化
- Git 仓库初始化，`.gitignore` 配置（排除 models/、outputs/、data/、音频文件）
- `.env.example` 模板（DeepSeek API Key + 本地路径配置）
- 目录结构搭建（services/api、apps/web、docs）

#### FastAPI 后端 (services/api)
- `config.py` — pydantic-settings 配置管理，读取 `.env` 文件
- `db.py` — SQLAlchemy + SQLite，支持环境变量覆盖数据库路径
- `models.py` — Job ORM 模型（id/title/lyrics/style_prompt/instrumental/status/audio_path/error/时间戳）
- `schemas.py` — Pydantic 请求/响应模型
- `main.py` — FastAPI 应用入口，lifespan 管理 DB 初始化和 Worker 启停
- CORS 中间件，仅允许 `http://localhost:3000`

#### API 端点 (4 个 Router)

| Router | 端点 | 说明 |
|--------|------|------|
| `/health` | GET | 健康检查 |
| `/arrange` | POST | DeepSeek 编排 → 结构化 JSON（歌词+风格） |
| `/generate` | POST | 创建出歌任务，入队等 Worker 处理 |
| `/jobs/{id}` | GET | 查询任务状态与结果 |
| `/library` | GET | 曲库列表，支持 `?status=` 筛选 |
| `/audio/{filename}` | GET | 本地音频文件服务 |

#### DeepSeek 编排服务
- 系统提示词 `arrange_system.txt` — 严格的 JSON 输出格式 + 风格维度约束
- `deepseek_arrange.py` — DeepSeek API 调用封装，支持：
  - `response_format: json_object` 结构化输出
  - JSON 提取（处理 markdown 代码块包裹）
  - Schema 校验（必填字段 + 风格子字段）
  - 失败自动重试 1 次

#### ACE-Step Runner 接口
- 三种后端模式：`subprocess`（默认）/ `api` / `import`
- `subprocess` 模式：通过 CLI 调用 ACE-Step，解耦最彻底
- 当前因安全策略未拉取 ACE-Step 仓库，需手动安装

#### Worker 队列
- 基于 `asyncio.Queue` 的内存任务队列
- lifespan 管理 Worker 生命周期
- 自动更新 Job 状态：`queued` → `running` → `succeeded`/`failed`

#### 测试 (12 tests, 全部通过)
- `test_health.py` — 健康检查
- `test_jobs_api.py` — 任务创建/查询/404
- `test_arrange_schema.py` — 编排 JSON 校验 + /arrange 端点 mock 测试
- `test_library.py` — 曲库列表/筛选/空库
- `conftest.py` — 共享 fixtures（临时 SQLite + mock DeepSeek key）

### 待办

#### 阻塞项
- [ ] **安装 CUDA PyTorch** — 当前环境为 CPU 版，需：
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```
- [ ] **安装 ACE-Step 1.5** — 克隆仓库并 pip install -e .
- [ ] **配置 DeepSeek API Key** — 在 `.env` 中填入 `DEEPSEEK_API_KEY`
- [ ] **验证 ACE-Step 推理** — 用真实歌词生成一首歌确认链路通

#### P1（下一阶段）
- [ ] Next.js 应用脚手架 + 导航
- [ ] 编曲房页面 (/studio)
- [ ] 专业创作页面 (/create)
- [ ] 前后端联调（编曲→创作→出歌→曲库 闭环）

#### P2
- [ ] 曲库页面 (/library)
- [ ] 风格标签库
- [ ] 体验打磨

### 设计决策记录

1. **仅绑定 127.0.0.1** — 不上线、不部署，纯粹本地工具
2. **SQLite 而非 PostgreSQL** — 单机单用户，零运维
3. **内存队列而非 Redis/RabbitMQ** — MVP 阶段单卡串行推理，不需要外部依赖
4. **ACE-Step subprocess 模式优先** — 环境隔离好，ACE-Step 升级不影响 API
5. **DeepSeek 作为唯一云依赖** — 编排用云端 LLM，出歌完全本地
6. **Python 3.11 独立 conda 环境给 ACE-Step** — 兼容性考虑（ACE-Step 不支持 3.13）
