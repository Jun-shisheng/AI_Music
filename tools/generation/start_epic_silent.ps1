# 静默后台启动英文史诗战斗曲批量生成（YuE2，6 小时预算）。
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
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\generation\start_epic_silent.ps1

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

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $LogDir "epic_run_$stamp.log"
$err = Join-Path $LogDir "epic_run_$stamp.err.log"

$pyArgs = @(
    '-I', '-u', $Driver,
    '--inputs',  'inputs\2026-10-14_epic_en',
    '--out-dir', 'outputs\2026-10-14_epic_en',
    '--hours', '6',
    '--candidates', '2',
    '--retries', '1',
    '--nar-query-chunk', '1024',
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
