$ErrorActionPreference = "Stop"
$root = "D:\Model\AI_Music"
$log = Join-Path $root "logs\yue2_batch_20261022.log"
$status = Join-Path $root "logs\yue2_batch_20261022_status.json"
$python = Join-Path $root "vendor\YuE2\.venv\Scripts\python.exe"
$batch = Join-Path $root "tools\generation\yue2_batch_battle.py"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

function Write-Status($stage, $state, $note) {
    $obj = [ordered]@{ stage = $stage; state = $state; time = (Get-Date).ToString("s"); note = $note }
    $obj | ConvertTo-Json | Set-Content -Path $status -Encoding UTF8
}

function Run-Batch($name, $inputs, $out, $language, $seed) {
    $now = Get-Date
    if ($now.Hour -gt 17 -or ($now.Hour -eq 17 -and $now.Minute -ge 20)) {
        Write-Status $name "skipped" "deadline"
        return
    }
    Write-Status $name "running" $out
    "[$(Get-Date -Format 's')] START $name" | Out-File -FilePath $log -Append -Encoding utf8
    & $python $batch --inputs $inputs --out-dir $out --candidates 1 --retries 1 `
        --budget 12 --backend torch-eager --nar-query-chunk 2048 `
        --language $language --seed-base $seed --intro-bars 2 `
        --shape "split chorus:2" --shape "thin verse:1.5" --final-lift 2 `
        --early-accept *>> $log
    $code = $LASTEXITCODE
    "[$(Get-Date -Format 's')] END $name exit=$code" | Out-File -FilePath $log -Append -Encoding utf8
    Write-Status $name "finished" "exit=$code"
}

Write-Status "start" "running" "epic_en + mandarin_pop"
Run-Batch "epic_en" (Join-Path $root "inputs\_batch_20261022_en") `
    (Join-Path $root "outputs\2026-10-22_batch\epic_en") "en" 2200
Run-Batch "mandarin_pop" (Join-Path $root "inputs\_batch_20261022_zh") `
    (Join-Path $root "outputs\2026-10-22_batch\mandarin_pop") "zh" 2400
Write-Status "all" "finished" "done"
