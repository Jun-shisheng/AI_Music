# 2026-10-23 gap-fill batch.
#
# One OS process per title. On 2026-10-22 the batch ran every song inside a
# single process and the later titles OOM'd (02_unbowed, 07_crown_of_thorns,
# 08_we_who_remain) even though their plans were not longer than the ones that
# succeeded (221.9 s vs 237.4 s / 224.3 s). A fresh process per title gives a
# clean allocator, which is what the --only help text already recommends.
#
# Each title gets two stages: stage 1 is the 2026-10-22 recipe with the process
# memory fraction raised to 0.90; if no selected WAV appears, stage 2 drops to a
# 768 NAR chunk, a 220 s plan cap and a different seed base.
#
# Why the fraction: YuE2Pipeline pins the process to (budget - 2) GiB = 10 GiB
# on a 12 GiB card. On 2026-10-22 the renders that worked were 181-204 s and the
# ones that OOM'd were 201-238 s (02_unbowed 228.7/226.3, 07 233.7/238.3,
# 08 201.0), all with ~9.7 GiB in use -- i.e. the wall was the memory cap, not
# the card. The desktop holds only ~0.9 GiB, so 0.90 (~11.0 GiB) recovers the
# headroom. Capping plans at 195 s instead was tried first and rejected: the
# model plans 218-228 s for these briefs, so every seed was skipped.
#
# Silent, windowless: this script is meant to be launched with Start-Process.

$ErrorActionPreference = "Stop"
$root = "D:\Model\AI_Music"
$python = Join-Path $root "vendor\YuE2\.venv\Scripts\python.exe"
$batch = "tools/generation/yue2_batch_battle.py"
$logDir = Join-Path $root "logs"
$statusFile = Join-Path $logDir "yue2_batch_20261023_status.json"

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
    $outLog = Join-Path $logDir "yue2_20261023_${label}_${slug}.out.log"
    $errLog = Join-Path $logDir "yue2_20261023_${label}_${slug}.err.log"
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
    "[$(Get-Date -Format 's')] START $label/$slug seed=$seed chunk=$chunk cap=$cap retries=$retries" |
        Out-File -FilePath (Join-Path $logDir "yue2_batch_20261023.log") -Append -Encoding utf8
    $p = Start-Process -FilePath $python -ArgumentList $arg -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
    $p.WaitForExit()
    $ok = Test-Path (Join-Path $outDir ("selected\" + $slug + ".wav"))
    "[$(Get-Date -Format 's')] END $label/$slug exit=$($p.ExitCode) selected=$ok" |
        Out-File -FilePath (Join-Path $logDir "yue2_batch_20261023.log") -Append -Encoding utf8
    return $ok
}

function Run-Title($label, $inputs, $outDir, $slug, $lang, $seed, $chunk2, $cap2, $chunk1, $cap1, $retries) {
    Write-Status "$label/$slug" "running" "$outDir"
    $ok = Invoke-Title $label $inputs $outDir $slug $lang $seed $chunk1 $cap1 $retries
    if (-not $ok) {
        Write-Status "$label/$slug" "retry" "stage 2: chunk=$chunk2 cap=$cap2"
        $ok = Invoke-Title $label $inputs $outDir $slug $lang ($seed + 500) $chunk2 $cap2 $retries
    }
    if ($ok) {
        Write-Status "$label/$slug" "done" "selected"
    } else {
        Write-Status "$label/$slug" "failed" "both stages produced no selected take"
    }
}

$enInputs = Join-Path $root "inputs\_batch_20261022_en"
$enOut = Join-Path $root "outputs\2026-10-23_batch\epic_en"
$enTitles = @("02_unbowed", "07_crown_of_thorns", "08_we_who_remain",
              "09_tide_of_endings", "10_the_sky_we_burned",
              "11_vow_under_ruins", "12_rise_again")
$i = 0
foreach ($t in $enTitles) {
    Run-Title "epic_en" $enInputs $enOut $t "en" (2500 + $i) 768 220 1024 240 4
    $i++
}

$zhInputs = Join-Path $root "inputs\_batch_20261022_zh2"
$zhOut = Join-Path $root "outputs\2026-10-23_batch\mandarin_pop"
$zhTitles = @("17_stay_until_sunrise", "18_gravity_is_a_lie",
              "19_lavender_city_lights", "20_better_than_yesterday")
$i = 0
foreach ($t in $zhTitles) {
    Run-Title "mandarin_pop" $zhInputs $zhOut $t "zh" (2800 + $i) 768 220 1024 240 4
    $i++
}

Write-Status "all" "finished" "gap-fill batch done"
