@echo off
echo Starting AI Music Studio...
echo.

echo [1/3] Starting ACE-Step API (port 8765)...
start "ACE-Step" cmd /c "cd /d G:\AI_Music\vendor\ACE-Step-1.5 && uv run acestep-api --host 127.0.0.1 --port 8765"

echo [2/3] Starting FastAPI (port 8000)...
start "FastAPI" cmd /c "cd /d G:\AI_Music\services\api && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [3/3] Starting Next.js (port 3000)...
start "Next.js" cmd /c "cd /d G:\AI_Music\apps\web && npx next dev --port 3000"

echo.
echo All services launched. Open http://localhost:3000
echo.
echo Close this window or press Ctrl+C in each terminal to stop.
pause
