$ErrorActionPreference = "Continue"
$Root = "D:/Model/AI_Music"
$Py = "D:/Model/AI_Music/vendor/YuE2/.venv/Scripts/python.exe"
$Batch = "D:/Model/AI_Music/tools/generation/yue2_batch_battle.py"
$Log = "D:/Model/AI_Music/logs/epic_bgm2_20261028.log"
$Err = "D:/Model/AI_Music/logs/epic_bgm2_20261028.err.log"
$a = @("-I", $Batch,
  "--inputs", "D:/Model/AI_Music/inputs/2026-10-28_bgm2",
  "--out-dir", "D:/Model/AI_Music/outputs/2026-10-28_epic_batch/bgm",
  "--only", "12_break_the_formation",
  "--candidates", "1", "--retries", "1", "--seed-base", "7600",
  "--budget", "12", "--nar-query-chunk", "1024", "--gpu-mem-fraction", "0.90",
  "--max-plan-seconds", "240", "--no-diction", "--no-melody",
  "--lora-ar", "D:/Model/AI_Music/models/yue2/loras/tsfh-steps-from-hell/adapter-ar-195/lora.safetensors",
  "--lora-nar", "D:/Model/AI_Music/models/yue2/loras/tsfh-steps-from-hell/adapter-nar-194/lora.safetensors",
  "--ar-scale", "0.5", "--nar-scale", "0.375")
$p = Start-Process -FilePath $Py -ArgumentList $a -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError $Err -PassThru
$p.Id | Out-File -Encoding ascii "D:/Model/AI_Music/logs/epic_bgm2_20261028.pid"
Write-Output ("started pid " + $p.Id)
