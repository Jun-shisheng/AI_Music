$ErrorActionPreference = "Continue"
$root = "D:\Model\AI_Music"
$py = Join-Path $root "tools\vocal_repair\.venv\Scripts\python.exe"
$log = Join-Path $root "logs\more_covers_retake_20261020.log"
$err = Join-Path $root "logs\more_covers_retake_20261020.err.log"
$arguments = @(
  (Join-Path $root "tools\generation\retake_more_covers.py"),
  "--list", (Join-Path $root "inputs\workflows\2026-10-20_more_covers.json"),
  "--slugs", "07_signal_fire", "03_iron_lullaby", "08_we_who_remain",
  "--log-dir", (Join-Path $root "outputs\2026-10-20\_work\more_covers\retake_logs"),
  "--stop-at", "16:20"
)
Start-Process -FilePath $py -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $err
Write-Output "started"
