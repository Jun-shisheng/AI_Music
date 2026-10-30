# 2026-10-23 afternoon retakes: the titles whose automated gates failed.
#
# Targets (from the manifests, not from listening):
#   09_tide_of_endings        lyric coverage 0.528 (< 0.6)
#   19_lavender_city_lights   early peak, 4.6 dB lift
#   07_run_toward_the_light   flat, 3.3 dB lift
#   10_sing_tomorrow_bright   lyric coverage 0.378
#   11_glass_hearts_glow      lyric coverage 0.390
#   16_paper_planes_satellites lyric coverage 0.382
#
# Originals stay untouched: takes land in outputs/2026-10-23_retakes/ so the
# weekend review can put the two side by side.
#
# Hard stop: no new title starts after 16:40, so the last render ends by 17:00.

$ErrorActionPreference = "Stop"
$root = "D:\Model\AI_Music"
$python = Join-Path $root "vendor\YuE2\.venv\Scripts\python.exe"
$batch = "tools/generation/yue2_batch_battle.py"
$logDir = Join-Path $root "logs"
$statusFile = Join-Path $logDir "yue2_retakes_20261023_status.json"
$deadline = Get-Date "2026-10-23 16:40:00"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Status($stage, $state, $note) {
    $obj = [ordered]@{
        stage = $stage
        state = $state
        time  = (Get-Date).ToString("s")
        note  = $note
    }
    $obj | ConvertTo-Json | Set-Content -Path $statusFile -Encoding UTF8
}

function Invoke-Title($label, $inputs, $outDir, $slug, $lang, $seed, $chunk, $cap, $retries) {
    $outLog = Join-Path $logDir "yue2_retake_20261023_${label}_${slug}.out.log"
    $errLog = Join-Path $logDir "yue2_retake_20261023_${label}_${slug}.err.log"
    $arg = '-u "' + $batch + '"' +
           ' --inputs "' + $inputs + '"' +
           ' --out-dir "' + $outDir + '"' +
           ' --candidates 1 --retries ' + $retries +
           ' --budget 12 --backend torch-eager' +
           ' --nar-query-chunk ' + $chunk +
           ' --max-plan-seconds ' + $cap +
           ' --gpu-mem-fraction 0.90' +
           ' --language ' + $lang +
           ' --seed-base ' + $seed +
           ' --intro-bars 2' +
           ' --shape "split chorus:2"' +
           ' --shape "thin verse:1.5"' +
           ' --final-lift 2' +
           ' --early-accept' +
           ' --only "' + $slug + '"'
    "[$(Get-Date -Format 's')] START retake/$slug seed=$seed" |
        Out-File -FilePath (Join-Path $logDir "yue2_retakes_20261023.log") -Append -Encoding utf8
    $p = Start-Process -FilePath $python -ArgumentList $arg -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
    $p.WaitForExit()
    $ok = Test-Path (Join-Path $outDir ("selected\" + $slug + ".wav"))
    "[$(Get-Date -Format 's')] END retake/$slug selected=$ok" |
        Out-File -FilePath (Join-Path $logDir "yue2_retakes_20261023.log") -Append -Encoding utf8
    return $ok
}

function Run-Title($label, $inputs, $outDir, $slug, $lang, $seed, $chunk2, $cap2, $chunk1, $cap1, $retries) {
    if ((Get-Date) -ge $deadline) {
        Write-Status "retake/$slug" "skipped" "deadline 16:40"
        return
    }
    Write-Status "retake/$slug" "running" "$outDir"
    $ok = Invoke-Title $label $inputs $outDir $slug $lang $seed $chunk1 $cap1 $retries
    if (-not $ok) {
        Write-Status "retake/$slug" "retry" "stage 2: chunk=$chunk2 cap=$cap2"
        $ok = Invoke-Title $label $inputs $outDir $slug $lang ($seed + 500) $chunk2 $cap2 $retries
    }
    if ($ok) {
        Write-Status "retake/$slug" "done" "selected"
    } else {
        Write-Status "retake/$slug" "failed" "both stages produced no selected take"
    }
}

$enInputs = Join-Path $root "inputs\_batch_20261022_en"
$enOut = Join-Path $root "outputs\2026-10-23_retakes\epic_en"
Run-Title "epic_en" $enInputs $enOut "09_tide_of_endings" "en" 4000 768 220 1024 240 4

$zh1Inputs = Join-Path $root "inputs\_batch_20261022_zh"
$zh2Inputs = Join-Path $root "inputs\_batch_20261022_zh2"
$zhOut = Join-Path $root "outputs\2026-10-23_retakes\mandarin_pop"
$i = 0
foreach ($t in @("07_run_toward_the_light", "10_sing_tomorrow_bright",
                 "11_glass_hearts_glow", "16_paper_planes_satellites",
                 "19_lavender_city_lights")) {
    $inputs = if ($t -eq "07_run_toward_the_light") { $zh1Inputs } else { $zh2Inputs }
    Run-Title "mandarin_pop" $inputs $zhOut $t "zh" (4100 + $i * 37) 768 220 1024 240 3
    $i++
}

Write-Status "all" "finished" "retake batch done"
