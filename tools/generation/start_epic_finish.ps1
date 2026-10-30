# 静默启动「补跑 10-12 首 → 自动打包 → 自动上传仓库」这条链。
#
# 用户要求：关机前跑完就自动上传，没跑完就算了（生成可续跑，不会重做已完成的）。
# 链式顺序写在 finish_epic.bat 里，这里只负责把它 detach 出去、重定向日志、并防止
# 重复启动（两个 YuE2 进程会把 12GB 显存撑爆）。
#
# 本文件必须保存为 UTF-8 **带 BOM**：PS 5.1 会把无 BOM 的 UTF-8 当 GBK 读，中文注释
# 解码时吞掉换行，把下一行代码并进注释（2026-10-09 的 start_batch_silent.ps1 就这么
# 挂过）。请勿用会去掉 BOM 的编辑器重存。
#
# 用法:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\generation\start_epic_finish.ps1

$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Bat    = Join-Path $Root 'tools\generation\finish_epic.bat'
$LogDir = Join-Path $Root 'logs'

if (-not (Test-Path $Bat)) { Write-Output "ERROR: chain not found at $Bat"; exit 1 }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*run_epic_yue2.py*' -or $_.CommandLine -like '*yue2_batch_battle.py*' }
if ($running) {
    Write-Output "ALREADY_RUNNING pid=$($running.ProcessId -join ',')"
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $LogDir "epic_finish_$stamp.log"
$err = Join-Path $LogDir "epic_finish_$stamp.err.log"

Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/c', "`"$Bat`"" `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $out `
    -RedirectStandardError  $err

Start-Sleep -Seconds 6
$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -like '*run_epic_yue2.py*' }
if ($p) { Write-Output "STARTED pid=$($p.ProcessId -join ',') log=$out" }
else    { Write-Output "not confirmed yet; check $out and $err" }
