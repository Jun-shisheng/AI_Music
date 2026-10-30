$ErrorActionPreference = "Continue"
$root = "D:\Model\AI_Music"
$py = Join-Path $root "tools\vocal_repair\.venv\Scripts\python.exe"
$log = Join-Path $root "logs\more_covers_batch_20261020.log"
$err = Join-Path $root "logs\more_covers_batch_20261020.err.log"
$arguments = @(
  (Join-Path $root "tools\generation\cover_more_batch.py"),
  "--list", (Join-Path $root "inputs\workflows\2026-10-20_more_covers.json"),
  "--status", (Join-Path $root "outputs\2026-10-20\_work\more_covers\status.json"),
  "--stop-at", "16:30",
  "--min-free-mib", "6000"
)
Start-Process -FilePath $py -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err
Write-Output "started"
