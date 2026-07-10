# 本地 AI 音乐（编曲优先）

在本地复刻「教父音乐 / Suno」类产品的编曲与出歌核心体验。

## 目标

- **编曲房**: 要求/歌词 → DeepSeek → 结构化歌词 + 风格标签
- **专业创作页**: 结构标签、流派/情绪/乐器/人声/速度 → ACE-Step 1.5 → 音频
- **本地曲库**: 任务列表、播放、下载（无登录）

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

## 快速启动

### 1. 安装 API 依赖

```bash
cd services/api
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 3. 启动 API 服务

```bash
cd services/api
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. 安装 ACE-Step 1.5（出歌引擎）

ACE-Step 需要独立安装（Python 3.11-3.12，推荐使用 conda 环境）：

```bash
# 创建 conda 环境（本项目已创建 acestep 环境）
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

配置 API 调用 ACE-Step：
```bash
# .env 中设置
ACESTEP_BACKEND=subprocess    # subprocess | api | import
```

### 5. 启动前端（P1 阶段）

```bash
cd apps/web
npm install && npm run dev
```

## 技术栈

| 层 | 选型 |
|----|------|
| 前端 | Next.js + TypeScript |
| 后端 | FastAPI + Python 3.11+ |
| 编排 | DeepSeek API |
| 音乐模型 | ACE-Step 1.5 (MIT) |
| 存储 | SQLite + outputs/ |

## 运行测试

```bash
cd services/api
python -m pytest tests/ -v
```

## 许可证

本项目代码仅供学习研究使用。使用的开源模型版权归各自作者所有。
