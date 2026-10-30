@echo off
setlocal
set "TASK_ROOT=%~dp0"
set "TASK_PYTHON=%TASK_ROOT%tools\vocal_repair\.venv\Scripts\python.exe"
if not exist "%TASK_PYTHON%" set "TASK_PYTHON=python"
"%TASK_PYTHON%" "%TASK_ROOT%tools\music_workflows.py" %*
exit /b %errorlevel%
