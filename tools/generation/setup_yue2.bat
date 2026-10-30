@echo off
rem Build the YuE2 inference environment. Run detached (window hidden):
rem   powershell -NoProfile -Command "Start-Process cmd -ArgumentList '/c','D:\Model\AI_Music\tools\generation\setup_yue2.bat' -WindowStyle Hidden -RedirectStandardOutput ... "
rem ASCII-only on purpose: a BOM-less .ps1 with non-ASCII comments is mis-decoded
rem by PowerShell 5.1 and silently breaks the script. Keep .bat files ASCII too.
setlocal

set VENV=D:\Model\AI_Music\vendor\YuE2\.venv
set WHL=D:\Model\AI_Music\models\yue2\YuE2-3B\yue2_infer-0.1.3-py3-none-any.whl
set TUNA=https://pypi.tuna.tsinghua.edu.cn/simple

if not exist "%VENV%\Scripts\python.exe" (
    echo [setup] creating venv
    py -3.11 -m venv "%VENV%"
)

echo [setup] upgrading pip
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip -i %TUNA%

echo [setup] installing torch 2.10.0 (the version yue2-infer pins)
"%VENV%\Scripts\python.exe" -m pip install -i %TUNA% torch==2.10.0

echo [setup] installing yue2-infer wheel
"%VENV%\Scripts\python.exe" -m pip install -i %TUNA% "%WHL%"

echo [setup] import check
"%VENV%\Scripts\python.exe" -c "import torch, yue2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo [setup] DONE exit=%ERRORLEVEL%
endlocal
