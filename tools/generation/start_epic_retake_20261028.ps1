$ErrorActionPreference = "Continue"
$Root = "D:/Model/AI_Music"
$Py = "D:/Model/AI_Music/vendor/YuE2/.venv/Scripts/python.exe"
$Script = "D:/Model/AI_Music/tools/generation/run_epic_batch_20261028.py"
$Log = "D:/Model/AI_Music/logs/epic_retake_20261028_driver.log"
$Err = "D:/Model/AI_Music/logs/epic_retake_20261028_driver.err.log"
$a = @("-I", $Script, "--only", "01_the_ember,06_empty_house", "--max-plan-seconds", "210", "--intro-bars", "2", "--retries", "2", "--seed-offset", "300")
$p = Start-Process -FilePath $Py -ArgumentList $a -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $Err -PassThru
$p.Id | Out-File -Encoding ascii "D:/Model/AI_Music/logs/epic_retake_20261028.pid"
Write-Output ("started pid " + $p.Id)
