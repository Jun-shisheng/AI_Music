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
| Python | 3.13 |
| PyTorch | 需 CUDA 版本 |

## 快速启动

### 1. 安装依赖

```bash
# 后端
cd services/api
pip install -r requirements.txt

# 安装 CUDA 版 PyTorch（若当前为 CPU 版）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. 下载模型

```bash
# ACE-Step 1.5 权重下载到 models/ace-step-1.5
# 详见: https://ace-step.github.io/ace-step-v1.5.github.io/
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 4. 启动服务

```bash
# 终端 1: API
cd services/api
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# 终端 2: 前端（P1 阶段）
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

## 许可证

本项目代码仅供学习研究使用。使用的开源模型版权归各自作者所有。
