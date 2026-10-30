$ErrorActionPreference = "Continue"
$Root = "D:/Model/AI_Music"
$Py = Join-Path $Root "vendor/YuE2/.venv/Scripts/python.exe"
$Script = Join-Path $Root "tools/generation/run_epic_batch_20261028.py"
$Log = Join-Path $Root "logs/epic_fill_20261028_driver.log"
$Err = Join-Path $Root "logs/epic_fill_20261028_driver.err.log"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
$args = @("-I", $Script, "--only", "02_echo,06_empty,08_the,09_wind,10_stand", "--max-plan-seconds", "240", "--intro-bars", "2", "--retries", "2", "--seed-offset", "100")
$p = Start-Process -FilePath $Py -ArgumentList $args -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $Err -PassThru
$p.Id | Out-File -Encoding ascii (Join-Path $Root "logs/epic_fill_20261028.pid")
Write-Output ("started pid " + $p.Id)
