@echo off
setlocal enabledelayedexpansion

title AI Music Studio

:: Locate the repository from this script's own path, so a fresh clone works
:: wherever it was checked out. Override with AI_MUSIC_ROOT if you keep the
:: weights somewhere else. (This used to be hardcoded to one machine's drive.)
if not defined AI_MUSIC_ROOT set "AI_MUSIC_ROOT=%~dp0"
if "%AI_MUSIC_ROOT:~-1%"=="\" set "AI_MUSIC_ROOT=%AI_MUSIC_ROOT:~0,-1%"
set "ROOT=%AI_MUSIC_ROOT%"
set "ACESTEP_DIR=%ROOT%\vendor\ACE-Step-1.5"
set "ACESTEP_VENV=%ACESTEP_DIR%\.venv\Scripts\python.exe"
set "CHECKPOINTS=%ACESTEP_DIR%\checkpoints"
set "API_PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not exist "%API_PYTHON%" set "API_PYTHON=%ROOT%\services\api\.venv\Scripts\python.exe"

echo.
echo ============================================================
echo   AI Music Studio - Local Music Workbench
echo ============================================================
echo.

:: ---- Validate prerequisites before patching, downloading, or starting ----

echo [Pre] Checking local prerequisites...
if not exist "%ACESTEP_DIR%\acestep\api_server.py" (
    echo [ERROR] ACE-Step checkout not found at "%ACESTEP_DIR%".
    echo [ERROR] Install ACE-Step as described in README.md before starting.
    exit /b 1
)
if not exist "%ACESTEP_VENV%" (
    echo [ERROR] ACE-Step virtual environment not found: "%ACESTEP_VENV%".
    echo [ERROR] Create its virtual environment and install its dependencies first.
    exit /b 1
)
if not exist "%API_PYTHON%" (
    echo [ERROR] API virtual environment not found in "%ROOT%\.venv" or "%ROOT%\services\api\.venv".
    echo [ERROR] Create the API environment and install services\api\requirements.txt first.
    exit /b 1
)
"%ACESTEP_VENV%" -c "import uvicorn, fastapi" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ACE-Step Python environment is unusable or is missing uvicorn/fastapi.
    exit /b 1
)
"%API_PYTHON%" -c "import uvicorn, fastapi" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] API Python environment is unusable or is missing uvicorn/fastapi.
    exit /b 1
)
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not available on PATH. Install it before starting.
    exit /b 1
)
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm is not available on PATH. Install it before starting.
    exit /b 1
)
if not exist "%ROOT%\apps\web\node_modules\next\dist\bin\next" (
    echo [ERROR] Frontend dependencies are missing. Run npm install in "%ROOT%\apps\web" first.
    exit /b 1
)
pushd "%ROOT%\apps\web"
if errorlevel 1 exit /b 1
call npm ls --depth=0 >nul 2>&1
set "FRONTEND_CHECK=%ERRORLEVEL%"
popd
if not "%FRONTEND_CHECK%"=="0" (
    echo [ERROR] Frontend dependencies are incomplete. Run npm install in "%ROOT%\apps\web" first.
    exit /b 1
)

echo [Pre] Checking ports 8765, 8000, 3000...
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; try { $listeners=@(Get-NetTCPConnection -State Listen -ErrorAction Stop | Where-Object { $_.LocalPort -in @(8765,8000,3000) }); foreach ($listener in $listeners) { $owner=Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue; Write-Host ('[ERROR] Port {0} is in use by PID {1} ({2}) on {3}.' -f $listener.LocalPort,$listener.OwningProcess,$owner.ProcessName,$listener.LocalAddress) }; if ($listeners.Count -gt 0) { exit 1 }; exit 0 } catch { Write-Host ('[ERROR] Unable to check listening ports: ' + $_.Exception.Message); exit 1 }"
if errorlevel 1 (
    echo [ERROR] Startup cancelled. Resolve port conflicts before retrying.
    exit /b 1
)
echo [OK] Prerequisites ready and ports available.

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
echo [ERROR] ACE-Step startup timed out. Check its terminal before retrying.
exit /b 1
:acestep_ready

:: ---- Start FastAPI (port 8000) ----

echo.
echo [2/3] Starting FastAPI backend (127.0.0.1:8000)...

(
echo @echo off
echo cd /d "%ROOT%\services\api"
echo "%API_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
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
echo [ERROR] FastAPI startup timed out. Check its terminal before retrying.
exit /b 1
:fastapi_ready

:: ---- Start Next.js (port 3000) ----

echo.
echo [3/3] Starting frontend (http://127.0.0.1:3000)...

(
echo @echo off
echo cd /d "%ROOT%\apps\web"
echo node node_modules\next\dist\bin\next dev --hostname 127.0.0.1 --port 3000
echo if errorlevel 1 pause
) > "%TEMP%\nextjs_launcher.bat"
start "Next.js" "%TEMP%\nextjs_launcher.bat"

set "TRIES=0"
:wait_next
timeout /t 2 /nobreak >nul
set /a TRIES+=1
powershell -Command "try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:3000' -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Next.js ready
    goto :next_ready
)
if %TRIES% lss 15 goto :wait_next
echo [ERROR] Next.js startup timed out. Check its terminal before retrying.
exit /b 1
:next_ready

:: ---- Done ----

echo.
echo ============================================================
echo   All services started!
echo.
echo   ACE-Step API    http://127.0.0.1:8765
echo   FastAPI         http://127.0.0.1:8000/docs
echo   Frontend        http://127.0.0.1:3000
echo ============================================================
echo.
echo Press any key to open browser, or close this window to stop.
pause >nul
start http://127.0.0.1:3000
echo.
echo Browser opened. Close each terminal window to stop services.
echo Next startup will be faster (models cached).
echo.
pause
