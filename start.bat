@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

title AI Music Studio

set "ROOT=G:\AI_Music"
set "ACESTEP_DIR=%ROOT%\vendor\ACE-Step-1.5"
set "ACESTEP_VENV=%ACESTEP_DIR%\.venv\Scripts\python.exe"
set "CHECKPOINTS=%ACESTEP_DIR%\checkpoints"

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║         AI Music Studio - 本地音乐工作台          ║
echo ╚══════════════════════════════════════════════════╝
echo.

:: ── Phase 0: Check models ────────────────────────────────────────────────

echo [0/3] 检查模型权重...
set "MODELS_OK=0"
if exist "%CHECKPOINTS%\acestep-v15-turbo\model.safetensors" set "MODELS_OK=1"
if exist "%CHECKPOINTS%\acestep-v15-turbo\model.safetensors.index.json" set "MODELS_OK=1"
if exist "%CHECKPOINTS%\acestep-v15-turbo\diffusion_pytorch_model.safetensors" set "MODELS_OK=1"
if exist "%CHECKPOINTS%\acestep-v15-turbo\diffusion_pytorch_model.safetensors.index.json" set "MODELS_OK=1"

if "%MODELS_OK%"=="0" (
    echo.
    echo [模型下载] 模型权重未找到，开始下载 (~10GB)...
    echo [模型下载] 将自动选择 HuggingFace 或 ModelScope 镜像
    echo.
    cd /d "%ACESTEP_DIR%"
    call "%ACESTEP_VENV%" -m acestep.model_downloader --dir "%CHECKPOINTS%"
    if %ERRORLEVEL% neq 0 (
        echo [错误] 模型下载失败，请检查网络连接后重试
        pause
        exit /b 1
    )
    echo [模型下载] 完成！
    echo.
) else (
    echo [OK] 模型权重已就绪
)

:: ── Phase 1: ACE-Step API (port 8765) ────────────────────────────────────

echo.
echo [1/3] 启动 ACE-Step API (127.0.0.1:8765)...

set "ACESTEP_API_HOST=127.0.0.1"
set "ACESTEP_API_PORT=8765"
set "ACESTEP_NO_INIT=false"
set "ACESTEP_INIT_LLM=false"
set "ACESTEP_CHECKPOINTS_DIR=%CHECKPOINTS%"

start "ACE-Step API" cmd /c "cd /d "%ACESTEP_DIR%" && "%ACESTEP_VENV%" -m uvicorn acestep.api_server:app --host 127.0.0.1 --port 8765 --workers 1 --log-level info"

:: Wait for ACE-Step to be ready (health check + model loading can take 2-5 min)
echo [等待] ACE-Step 模型加载中（首次需 2-5 分钟）...
set "WAIT_COUNT=0"
:wait_acestep
timeout /t 5 /nobreak >nul
set /a WAIT_COUNT+=5
curl -s http://127.0.0.1:8765/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] ACE-Step API 就绪 (等待 %WAIT_COUNT%s^)
    goto :acestep_ready
)
if %WAIT_COUNT% lss 600 (
    if %WAIT_COUNT% equ 30 echo [等待] 首次加载需要下载/初始化模型，请耐心等待...
    goto :wait_acestep
)
echo [警告] ACE-Step 启动超时，继续启动其他服务...
:acestep_ready

:: ── Phase 2: FastAPI (port 8000) ──────────────────────────────────────────

echo.
echo [2/3] 启动 FastAPI 后端 (127.0.0.1:8000)...
start "FastAPI" cmd /c "cd /d "%ROOT%\services\api" && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

:: Wait for FastAPI
set "WAIT_COUNT=0"
:wait_fastapi
timeout /t 2 /nobreak >nul
set /a WAIT_COUNT+=2
curl -s http://127.0.0.1:8000/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] FastAPI 就绪
    goto :fastapi_ready
)
if %WAIT_COUNT% lss 30 goto :wait_fastapi
echo [警告] FastAPI 启动超时
:fastapi_ready

:: ── Phase 3: Next.js (port 3000) ──────────────────────────────────────────

echo.
echo [3/3] 启动前端 (http://localhost:3000)...
start "Next.js" cmd /c "cd /d "%ROOT%\apps\web" && npx next dev --port 3000"

:: Wait for Next.js
set "WAIT_COUNT=0"
:wait_next
timeout /t 2 /nobreak >nul
set /a WAIT_COUNT+=2
curl -s http://localhost:3000 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Next.js 就绪
    goto :next_ready
)
if %WAIT_COUNT% lss 30 goto :wait_next
echo [警告] Next.js 启动超时
:next_ready

:: ── Done ──────────────────────────────────────────────────────────────────

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║           全部服务已启动                           ║
echo ╠══════════════════════════════════════════════════╣
echo ║  ACE-Step API     http://127.0.0.1:8765           ║
echo ║  FastAPI 后端     http://127.0.0.1:8000/docs      ║
echo ║  前端工作台       http://localhost:3000            ║
echo ╚══════════════════════════════════════════════════╝
echo.
echo 按任意键打开浏览器，或关闭此窗口停止所有服务...

:: Open browser
pause >nul
start http://localhost:3000
echo.
echo 浏览器已打开。关闭各终端窗口即可停止服务。
echo 下次启动速度会更快（模型已缓存）。
echo.
pause
