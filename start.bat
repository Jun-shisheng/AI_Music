@echo off
setlocal enabledelayedexpansion

title AI Music Studio

set "ROOT=G:\AI_Music"
set "ACESTEP_DIR=%ROOT%\vendor\ACE-Step-1.5"
set "ACESTEP_VENV=%ACESTEP_DIR%\.venv\Scripts\python.exe"
set "CHECKPOINTS=%ACESTEP_DIR%\checkpoints"
set "API_PYTHON=%ROOT%\services\api\.venv\Scripts\python.exe"

:: Fall back to system python if API venv doesn't exist
if not exist "%API_PYTHON%" set "API_PYTHON=python"

echo.
echo ============================================================
echo   AI Music Studio - Local Music Workbench
echo ============================================================
echo.

:: ---- Clean stale processes ----

echo [Pre] Cleaning stale processes on ports 8765, 8000, 3000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8765.*LISTENING" 2^>nul') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING" 2^>nul') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3000.*LISTENING" 2^>nul') do taskkill /F /PID %%a 2>nul
echo [OK] Ports cleared.

:: ---- Apply patches to ACE-Step (one-time) ----

if exist "%ROOT%\patches\ace-step-int8-quantization.patch" (
    if exist "%ACESTEP_DIR%\.git" (
        cd /d "%ACESTEP_DIR%"
        git apply --check "%ROOT%\patches\ace-step-int8-quantization.patch" >nul 2>&1
        if errorlevel 1 (
            echo [Patch] INT8 quantization already applied
        ) else (
            git apply "%ROOT%\patches\ace-step-int8-quantization.patch" >nul 2>&1
            if errorlevel 1 (
                echo [WARN] Failed to apply INT8 quantization patch
            ) else (
                echo [Patch] INT8 quantization enabled for 8GB GPU
            )
        )
    )
)

:: ---- Check / Download Models ----

echo [0/3] Checking model weights...

set "MODELS_OK=0"
if exist "%CHECKPOINTS%\acestep-v15-turbo\model.safetensors" set "MODELS_OK=1"
if exist "%CHECKPOINTS%\acestep-v15-turbo\model.safetensors.index.json" set "MODELS_OK=1"
if exist "%CHECKPOINTS%\acestep-v15-turbo\diffusion_pytorch_model.safetensors" set "MODELS_OK=1"
if exist "%CHECKPOINTS%\acestep-v15-turbo\diffusion_pytorch_model.safetensors.index.json" set "MODELS_OK=1"

if "%MODELS_OK%"=="0" (
    echo.
    echo [Download] Models not found. Downloading ~10GB...
    echo [Download] Will auto-select HuggingFace or ModelScope mirror.
    echo.
    cd /d "%ACESTEP_DIR%"
    call "%ACESTEP_VENV%" -m acestep.model_downloader --dir "%CHECKPOINTS%"
    if errorlevel 1 (
        echo [ERROR] Model download failed. Check network and retry.
        pause
        exit /b 1
    )
    echo [Download] Done!
    echo.
) else (
    echo [OK] Model weights ready.
)

:: ---- Start ACE-Step API (port 8765) ----

echo.
echo [1/3] Starting ACE-Step API (127.0.0.1:8765)...

set "ACESTEP_API_HOST=127.0.0.1"
set "ACESTEP_API_PORT=8765"
set "ACESTEP_NO_INIT=false"
set "ACESTEP_INIT_LLM=false"
set "ACESTEP_CHECKPOINTS_DIR=%CHECKPOINTS%"
set "ACESTEP_TMPDIR=%ROOT%\outputs"

:: Write a launcher script for ACE-Step to avoid nested quote issues
(
echo @echo off
echo cd /d "%ACESTEP_DIR%"
echo set ACESTEP_API_HOST=127.0.0.1
echo set ACESTEP_API_PORT=8765
echo set ACESTEP_NO_INIT=false
echo set ACESTEP_INIT_LLM=false
echo set ACESTEP_CHECKPOINTS_DIR=%CHECKPOINTS%
echo set ACESTEP_TMPDIR=%ROOT%\outputs
echo "%ACESTEP_VENV%" -m uvicorn acestep.api_server:app --host 127.0.0.1 --port 8765 --workers 1 --log-level info
echo if errorlevel 1 pause
) > "%TEMP%\acestep_launcher.bat"
start "ACE-Step API" "%TEMP%\acestep_launcher.bat"

echo [Wait] ACE-Step loading models (first time 2-5 min)...

set "TRIES=0"
:wait_acestep
timeout /t 5 /nobreak >nul
set /a TRIES+=1
powershell -Command "try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:8765/health' -TimeoutSec 3 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set /a SECS=%TRIES%*5
    echo [OK] ACE-Step API ready ^(!SECS!s^)
    goto :acestep_ready
)
if %TRIES% lss 120 (
    if %TRIES% equ 6 echo [Wait] First startup may take a few minutes, please wait...
    goto :wait_acestep
)
echo [WARN] ACE-Step startup timeout, continuing anyway...
:acestep_ready

:: ---- Start FastAPI (port 8000) ----

echo.
echo [2/3] Starting FastAPI backend (127.0.0.1:8000)...

(
echo @echo off
echo cd /d "%ROOT%\services\api"
echo "%API_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
echo if errorlevel 1 pause
) > "%TEMP%\fastapi_launcher.bat"
start "FastAPI" "%TEMP%\fastapi_launcher.bat"

set "TRIES=0"
:wait_fastapi
timeout /t 2 /nobreak >nul
set /a TRIES+=1
powershell -Command "try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] FastAPI ready
    goto :fastapi_ready
)
if %TRIES% lss 15 goto :wait_fastapi
echo [WARN] FastAPI startup timeout - starting anyway
:fastapi_ready

:: ---- Start Next.js (port 3000) ----

echo.
echo [3/3] Starting frontend (http://localhost:3000)...

(
echo @echo off
echo cd /d "%ROOT%\apps\web"
echo npx next dev --port 3000
echo if errorlevel 1 pause
) > "%TEMP%\nextjs_launcher.bat"
start "Next.js" "%TEMP%\nextjs_launcher.bat"

set "TRIES=0"
:wait_next
timeout /t 2 /nobreak >nul
set /a TRIES+=1
powershell -Command "try { $r=Invoke-WebRequest -Uri 'http://localhost:3000' -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Next.js ready
    goto :next_ready
)
if %TRIES% lss 15 goto :wait_next
echo [WARN] Next.js startup timeout
:next_ready

:: ---- Done ----

echo.
echo ============================================================
echo   All services started!
echo.
echo   ACE-Step API    http://127.0.0.1:8765
echo   FastAPI         http://127.0.0.1:8000/docs
echo   Frontend        http://localhost:3000
echo ============================================================
echo.
echo Press any key to open browser, or close this window to stop.
pause >nul
start http://localhost:3000
echo.
echo Browser opened. Close each terminal window to stop services.
echo Next startup will be faster (models cached).
echo.
pause
