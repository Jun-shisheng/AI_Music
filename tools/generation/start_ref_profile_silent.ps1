# Profile the reference tracks AND our own take with the SAME instrument.
#
# Why both: Wildfire measured 2.88 vocal onsets/s in its soft windows, while our
# 01_hammer_and_hymn measured 1.16-1.27 WORDS/s via whisper. Those are different
# units from different methods and must never be compared directly. Running our
# own track through ref_profile.py gives the apples-to-apples number.
#
# Resumable: decode/separate/analyse results are cached per track under
# human_refs/refs/_work, so re-running only does what is missing.
#
# ASCII-only on purpose (PowerShell 5.1 reads a BOM-less UTF-8 .ps1 as GBK and a
# mis-decoded CJK comment swallows the next line of code).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\generation\start_ref_profile_silent.ps1

$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Py     = Join-Path $Root 'tools\vocal_repair\.venv\Scripts\python.exe'
$Tool   = Join-Path $Root 'tools\generation\ref_profile.py'
$LogDir = Join-Path $Root 'logs'

if (-not (Test-Path $Py))   { Write-Output "ERROR: vocal_repair venv python not found at $Py"; exit 1 }
if (-not (Test-Path $Tool)) { Write-Output "ERROR: tool not found at $Tool"; exit 1 }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*ref_profile.py*' }
if ($running) {
    Write-Output "ALREADY_RUNNING pid=$($running.ProcessId -join ',')"
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $LogDir "ref_profile_$stamp.log"
$err = Join-Path $LogDir "ref_profile_$stamp.err.log"

# One python process, ours first then the whole reference set (ref_profile.py
# profiles --track entries before --all, so the comparison is available early).
# An earlier version chained two commands through a generated inner .ps1 and the
# argument passing silently failed, launching both jobs at once; one process with
# both jobs on its own command line has no such seam.
$pyArgs = @(
    '-I', $Tool,
    '--track', 'outputs\2026-10-15_epic_en\selected\01_hammer_and_hymn_raw.wav',
    '--all',
    '--whisper',
    '--json', 'logs\ref_all.json',
    '--report', 'human_refs\refs\_profiles.md'
)

Start-Process -FilePath $Py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $out `
    -RedirectStandardError  $err

Start-Sleep -Seconds 10
$p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -like '*ref_profile.py*' }
if ($p) { Write-Output "STARTED pid=$($p.ProcessId -join ',') log=$out" }
else    { Write-Output "NOT STARTED YET; check $err" }
