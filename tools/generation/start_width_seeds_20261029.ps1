$ErrorActionPreference = "Continue"
$Root = "D:/Model/AI_Music"
$Py = Join-Path $Root "vendor/YuE2/.venv/Scripts/python.exe"
$Script = Join-Path $Root "tools/generation/run_width_seeds_20261029.py"
$Log = Join-Path $Root "logs/width_seeds_20261029_driver.log"
$Err = Join-Path $Root "logs/width_seeds_20261029_driver.err.log"
$PidFile = Join-Path $Root "logs/width_seeds_20261029.pid"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
$p = Start-Process -FilePath $Py -ArgumentList @("-I", $Script) -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $Err -PassThru
$p.Id | Out-File -Encoding ascii $PidFile
Write-Output ("started pid " + $p.Id)
