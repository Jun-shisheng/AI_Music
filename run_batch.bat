@echo off
setlocal
title AI Music - Batch Generation

:: Run the tuned batch generator in its OWN console window.
::
:: Why a separate window: when the generator runs as a background job inside an
:: editor/agent session, killing that job leaves the python child orphaned while
:: it still holds ~11 GB of model weights, which then starves the next run. A
:: standalone console has no such parent and is not subject to that reaper.
::
:: Already-generated songs are skipped, so this is safe to run repeatedly and
:: resumes wherever it left off. A take whose ending degenerates is retried
:: automatically with a new seed.
::
:: English titles (11-20) are scheduled first via --priority, then the rest; the
:: per-song seed/temperature stay tied to each title's sorted position, so the
:: order of the run does not change any song's parameters.
::
:: For a silent, windowless launch (log to logs\batch_*.log) use instead:
::   powershell -NoProfile -ExecutionPolicy Bypass -File tools\generation\start_batch_silent.ps1

set "ROOT=%~dp0"
set "PYTHON=%ROOT%vendor\ACE-Step-1.5\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] venv python not found at:
    echo         %PYTHON%
    pause
    exit /b 1
)

cd /d "%ROOT%"

echo ============================================================
echo   Generating songs -^> outputs\final
echo   (skip existing, auto-retry degenerate endings)
echo ============================================================
echo.

"%PYTHON%" -u tools\generation\batch_tuned.py --inputs inputs --out-dir outputs\final --seconds 210 --priority "^1[1-9]_|^20_"

echo.
echo ============================================================
echo   Done. Press any key to close.
echo ============================================================
pause >nul
