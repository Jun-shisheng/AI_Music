# 本地 AI 音乐（编曲优先）

**仅限本地运行。** 不做公网部署、不做用户体系、不做商业化。

在本地复刻「教父音乐 / Suno」类产品的编曲与出歌核心体验。

## 目标

- **编曲房**: 要求/歌词 → DeepSeek → 结构化歌词 + 风格标签
- **专业创作页**: 结构标签、流派/情绪/乐器/人声/速度 → ACE-Step 1.5 → 音频
- **本地曲库**: 任务列表、播放、下载（无登录）

## 运行原则

- 所有服务绑定 `127.0.0.1`，仅本机访问
- 不开放外网端口，不走公网 DNS
- 音频文件、数据库全部落本地磁盘
- 无账号注册、无云端同步、无遥测

## 硬件要求

| 配置 | 说明 |
|------|------|
| GPU | NVIDIA GPU，≥8GB 显存（理想 12GB+） |
| 磁盘 | ≥30GB（模型权重 ~10GB + 依赖 + 音频输出） |
| OS | Windows / Linux |

## 已验证环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8188 MiB |
| Python | 3.13 (API) / 3.11 (ACE-Step conda env) |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/arrange` | DeepSeek 编排（要求→结构化歌词+风格） |
| POST | `/generate` | 提交出歌任务 |
| GET | `/jobs/{id}` | 查询任务状态 |
| GET | `/library` | 曲库列表（支持 `?status=` 筛选） |
| GET | `/audio/{filename}` | 播放/下载音频文件 |

全部仅响应 `127.0.0.1`，不绑定外网。

## 快速启动

### 1. 安装 API 依赖

```bash
cd services/api
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY（仅用于 DeepSeek 云端 API 调用）
# 其他变量保持默认即可
```

### 3. 启动 API 服务

```bash
cd services/api
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API 启动后访问 http://127.0.0.1:8000/docs 查看 Swagger 文档。

### 4. 安装 ACE-Step 1.5（出歌引擎）

ACE-Step 需要独立安装（Python 3.11-3.12，推荐使用 conda 环境）：

```bash
conda create -n acestep python=3.11 -y
conda activate acestep

# 安装 CUDA PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 克隆并安装 ACE-Step
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5
pip install -e .

# 首次运行会自动下载模型（~10GB）
uv run acestep --init_service true
```

配置 API 调用 ACE-Step（在 `.env` 中设置）：
```bash
ACESTEP_BACKEND=subprocess    # subprocess | api | import
```

### 5. 启动前端（P1 阶段，待开发）

```bash
cd apps/web
npm install && npm run dev
# 访问 http://127.0.0.1:3000
```

## 数据流

```
浏览器 (127.0.0.1:3000)
    │
    ▼
FastAPI (127.0.0.1:8000)
    ├── POST /arrange  ──→ DeepSeek API (云端，仅编排)
    ├── POST /generate ──→ Worker 队列 ──→ ACE-Step 子进程 (本机 GPU)
    ├── GET  /library  ──→ SQLite (data/app.db)
    └── GET  /audio/*  ──→ outputs/*.wav (本地磁盘)
```

## 技术栈

| 层 | 选型 | 说明 |
|----|------|------|
| 前端 | Next.js + TypeScript | P1 阶段 |
| 后端 | FastAPI + Python | 已完成 |
| 编排 | DeepSeek API | 仅此一项依赖外部云服务 |
| 音乐模型 | ACE-Step 1.5 (MIT) | 本机 GPU 推理 |
| 存储 | SQLite + outputs/ | 全部本地文件 |

## 运行测试

```bash
cd services/api
python -m pytest tests/ -v
```

## 项目结构

```
G:/AI_Music/
├── README.md
├── .env.example              # 环境变量模板
├── .gitignore
├── data/                     # SQLite 数据库 (gitignored)
├── outputs/                  # 生成的音频文件 (gitignored)
├── models/                   # 模型权重 (gitignored)
├── services/
│   └── api/                  # FastAPI 后端
│       ├── app/
│       │   ├── main.py       # 应用入口
│       │   ├── config.py     # 配置 (pydantic-settings)
│       │   ├── db.py         # SQLite (SQLAlchemy)
│       │   ├── models.py     # Job ORM
│       │   ├── schemas.py    # 请求/响应模型
│       │   ├── routers/      # API 路由
│       │   ├── services/     # 业务逻辑
│       │   └── prompts/      # LLM 系统提示词
│       └── tests/            # pytest (12 tests)
├── apps/
│   └── web/                  # Next.js 前端 (待开发)
└── docs/                     # 设计文档
```

## 许可证

本项目代码仅供学习研究使用。使用的开源模型版权归各自作者所有。
