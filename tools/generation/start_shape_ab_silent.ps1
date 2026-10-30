# Render one controlled A/B take from a density-shaped plan (2026-10-16).
#
# Purpose: prove that editing the symbolic plan's note density actually changes
# what the singer does. Same brief, same seed (1200), same everything as the
# delivered 01_hammer_and_hymn -- the only difference is that the plan was run
# through plan_shape.py (verse --reduce 2, chorus --split 2).
#
# ASCII-only on purpose: PowerShell 5.1 reads a BOM-less UTF-8 .ps1 as GBK and a
# mis-decoded CJK comment swallows the newline after it, merging the next line of
# code into the comment. Keeping this file ASCII removes that whole failure mode.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\generation\start_shape_ab_silent.ps1

$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # tools\generation -> repo root
$Py     = Join-Path $Root 'vendor\YuE2\.venv\Scripts\python.exe'
$Tool   = Join-Path $Root 'tools\generation\yue2_batch_battle.py'
$Plan   = Join-Path $Root 'outputs\_shape_ab\01_hammer_and_hymn_shaped.abc'
$LogDir = Join-Path $Root 'logs'

if (-not (Test-Path $Py))   { Write-Output "ERROR: YuE2 venv python not found at $Py"; exit 1 }
if (-not (Test-Path $Tool)) { Write-Output "ERROR: tool not found at $Tool"; exit 1 }
if (-not (Test-Path $Plan)) { Write-Output "ERROR: shaped plan not found at $Plan (run plan_shape.py first)"; exit 1 }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Never start a second YuE2 process: two of them blow the 12 GB card.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*yue2_batch_battle.py*' -or $_.CommandLine -like '*run_epic_yue2.py*' }
if ($running) {
    Write-Output "ALREADY_RUNNING pid=$($running.ProcessId -join ',')"
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $LogDir "shape_ab_$stamp.log"
$err = Join-Path $LogDir "shape_ab_$stamp.err.log"

$pyArgs = @(
    '-I', '-u', $Tool,
    '--inputs',   'inputs\2026-10-15_epic_en',
    '--out-dir',  'outputs\_shape_ab',
    '--only',     '01_hammer_and_hymn',
    '--plan-file','outputs\_shape_ab\01_hammer_and_hymn_shaped.abc',
    '--seed-base','1200',
    '--candidates','1',
    '--nar-query-chunk', '1024',
    '--budget',   '12'
)

Start-Process -FilePath $Py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $out `
    -RedirectStandardError  $err

Start-Sleep -Seconds 8
$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -like '*yue2_batch_battle.py*' }
if ($p) { Write-Output "STARTED pid=$($p.ProcessId -join ',') log=$out" }
else    { Write-Output "FAILED to start; see $err" }
