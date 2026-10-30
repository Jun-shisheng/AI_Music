# Launch the one-day density-fix run, windowless and detached.
#
# The chain itself is run_density_chain.ps1 in this same folder; this script only
# guards against a second run and starts it with Start-Process, because a Git
# Bash child (nohup / &) does not survive the shell being reaped on Windows and
# would leave an orphaned python holding the model weights.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\generation\start_density_silent.ps1
#
# ASCII only (PowerShell 5.1 mis-decodes a BOM-less UTF-8 .ps1 as GBK).

$ErrorActionPreference = 'Stop'

$Root  = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Chain = Join-Path $Root 'tools\generation\run_density_chain.ps1'
$LogDir = Join-Path $Root 'logs'

if (-not (Test-Path $Chain))  { Write-Output "ERROR: chain not found at $Chain"; exit 1 }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Never two YuE2 runs at once: they share 12 GB of VRAM and would both OOM.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*run_epic_yue2.py*' -or
                          $_.CommandLine -like '*yue2_batch_battle.py*' }
if ($running) {
    Write-Output "ALREADY_RUNNING pid=$($running.ProcessId -join ',')"
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $LogDir "density_chain_$stamp.log"
$err = Join-Path $LogDir "density_chain_$stamp.err.log"

Start-Process -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Chain) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $out `
    -RedirectStandardError  $err

Start-Sleep -Seconds 12
$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -like '*run_epic_yue2.py*' }
if ($p) { Write-Output "STARTED pid=$($p.ProcessId -join ',') log=$out" }
else    { Write-Output "not started yet; check $err" }
