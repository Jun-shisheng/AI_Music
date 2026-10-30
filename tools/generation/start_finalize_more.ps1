$ErrorActionPreference = "Continue"
$root = "D:\Model\AI_Music"
$py = Join-Path $root "tools\vocal_repair\.venv\Scripts\python.exe"
$log = Join-Path $root "logs\more_covers_finalize_20261020.log"
$err = Join-Path $root "logs\more_covers_finalize_20261020.err.log"
$arguments = @(
  (Join-Path $root "tools\generation\finalize_more_covers.py"),
  "--list", (Join-Path $root "inputs\workflows\2026-10-20_more_covers.json"),
  "--work-dir", (Join-Path $root "outputs\2026-10-20\_work\more_covers"),
  "--local-out", (Join-Path $root "outputs\2026-10-20\更多固定女声"),
  "--delivery-out", (Join-Path $root "outputs\2026-10-20\fixed_voice_more"),
  "--target-seconds", "210"
)
Start-Process -FilePath $py -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err
Write-Output "started"
