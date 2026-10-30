$ErrorActionPreference = "Continue"
$Root = "D:\Model\AI_Music"
$Py = Join-Path $Root "vendor\YuE2\.venv\Scripts\python.exe"
$Script = Join-Path $Root "tools\generation\run_zh_pop_20261026.py"
$Log = Join-Path $Root "logs\zh_battle12_20261026_driver.log"
$Err = Join-Path $Root "logs\zh_battle12_20261026_driver.err.log"
$PidFile = Join-Path $Root "logs\zh_battle12_20261026.pid"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
$p = Start-Process -FilePath $Py -ArgumentList @("-I", $Script, "--only", "12", "--candidates", "1") -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $Err -PassThru
$p.Id | Out-File -Encoding ascii $PidFile
Write-Output ("started pid " + $p.Id)
