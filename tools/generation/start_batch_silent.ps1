# 静默后台启动批量生成（不弹窗、不依赖调用方的进程树）。
#
# 为什么不用 bash 的 nohup / 后台 shell：Git Bash 在 Windows 上无法让子进程真正
# 脱离进程树，父 shell 被回收时 python 子进程会被孤立却仍占着 ~11GB 模型权重，
# 把下一次运行饿死。Start-Process 创建的是独立进程，不受该问题影响。
#
# 输出走日志文件（logs\batch_*.log），进度也写在 outputs\final\BATCH_STATUS.txt，
# 读回后在对话里汇报，全程不弹任何窗口。
#
# 用法:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\generation\start_batch_silent.ps1

$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # tools\generation -> 仓库根
$Py     = Join-Path $Root 'vendor\ACE-Step-1.5\.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'logs'

if (-not (Test-Path $Py))    { Write-Output "ERROR: venv python not found at $Py"; exit 1 }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# 已经在跑就别重复起（两次加载会抢显存）
$already = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*batch_tuned.py*' }
if ($already) {
    Write-Output "ALREADY_RUNNING pid=$($already.ProcessId -join ',')"
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $LogDir "batch_$stamp.log"
$err = Join-Path $LogDir "batch_$stamp.err.log"

$pyArgs = @(
    '-u', 'tools\generation\batch_tuned.py',
    '--inputs', 'inputs',
    '--out-dir', 'outputs\final',
    '--seconds', '210',
    '--priority', '^1[1-9]_|^20_'      # 英文曲（11-20）先跑，其余随后
)

Start-Process -FilePath $Py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $out `
    -RedirectStandardError  $err

Start-Sleep -Seconds 5
$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -like '*batch_tuned.py*' }
if ($p) { Write-Output "STARTED pid=$($p.ProcessId -join ',') log=$out" }
else    { Write-Output "FAILED to start; see $err" }
