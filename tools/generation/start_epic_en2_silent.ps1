# 静默后台启动 2026-10-15 英文史诗战斗曲批量生成（YuE2，6 小时预算）。
#
# 与 start_epic_silent.ps1（10-14 那批）的唯一区别是这一批吃的新规则：
#   --intro-bars 2  渲染前把计划里的器乐前奏砍到 2 小节（约 3 秒），
#                   对应「要么前奏几秒直接开唱」。参考曲也是这样开场的
#                   （明日方舟 Radiant 0:02、崩铁 Wildfire 0:03）。
#   --max-onset 20  whisper 量到的首句人声晚于 20 秒就丢片重摇；
#                   10-14 那批实测 15.5-32.3 秒。
#
# 为什么不弹窗、不用 bash 的 nohup：Git Bash 在 Windows 上无法让子进程真正脱离
# 进程树，父 shell 被回收时 python 子进程会被孤立却仍占着 ~9GB 模型权重，把下一次
# 运行饿死。Start-Process 创建的才是独立进程。
#
# 本文件必须保存为 UTF-8 **带 BOM**：PS 5.1 会把无 BOM 的 UTF-8 当 GBK 读，中文注释
# 解码时吞掉换行，把下一行代码并进注释（2026-10-09 的 start_batch_silent.ps1 就这么
# 挂过）。请勿用会去掉 BOM 的编辑器重存。
#
# 用法:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\generation\start_epic_en2_silent.ps1

$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # tools\generation -> 仓库根
$Py     = Join-Path $Root 'vendor\YuE2\.venv\Scripts\python.exe'
$Driver = Join-Path $Root 'tools\generation\run_epic_yue2.py'
$LogDir = Join-Path $Root 'logs'

if (-not (Test-Path $Py))     { Write-Output "ERROR: YuE2 venv python not found at $Py"; exit 1 }
if (-not (Test-Path $Driver)) { Write-Output "ERROR: driver not found at $Driver"; exit 1 }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# 已经在跑就别重复起：两个 YuE2 进程会把 12GB 显存撑爆
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*run_epic_yue2.py*' -or $_.CommandLine -like '*yue2_batch_battle.py*' }
if ($running) {
    Write-Output "ALREADY_RUNNING pid=$($running.ProcessId -join ',')"
    exit 0
}

# 显存自检：ComfyUI 常驻会占掉 10-11GB，先关掉再跑
$free = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -as [int]
if ($free -gt 2000) {
    Write-Output "WARNING: GPU already holds ${free} MiB; close ComfyUI first if this is it"
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $LogDir "epic_en2_$stamp.log"
$err = Join-Path $LogDir "epic_en2_$stamp.err.log"

$pyArgs = @(
    '-I', '-u', $Driver,
    '--inputs',  'inputs\2026-10-15_epic_en',
    '--out-dir', 'outputs\2026-10-15_epic_en',
    '--hours', '6',
    '--candidates', '2',
    '--retries', '1',
    '--nar-query-chunk', '1024',
    '--intro-bars', '2',
    '--max-onset', '20',
    '--min-duration', '120',
    '--max-duration', '210'
)

Start-Process -FilePath $Py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $out `
    -RedirectStandardError  $err

Start-Sleep -Seconds 8
$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -like '*run_epic_yue2.py*' }
if ($p) { Write-Output "STARTED pid=$($p.ProcessId -join ',') log=$out" }
else    { Write-Output "FAILED to start; see $err" }
