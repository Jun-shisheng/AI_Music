# One-day run: re-render the 12 English epic titles with the chorus density fix,
# then build the dated delivery folder (local only -- no git push).
#
# Two fixes, both measurement-backed, and nothing else:
#   The user's complaint was that a whole song runs at one rhythm ("前期明显缓慢，
#   高潮更加集中" in HOYO-MiX Wildfire, "我们的从头到尾都是一个节奏").
#
#   1. HOOK (--shape "hook chorus"). Measured with one instrument across 10 vocal
#      references, the loud sections of a real song resemble each other at chroma
#      cosine 0.978 (median) -- a chorus is the same hook coming back. Ours
#      measured 0.902, below every reference: the plan writes each chorus a NEW
#      melody over the same chords, so nothing reads as a chorus. The operator
#      copies the first chorus's melody (chords included) onto the later ones,
#      tiling the motif when bar counts differ and matching the target register
#      in whole octaves. Lyrics are fed separately, so "the final chorus must be
#      new lyrics" still holds.
#
#   2. DENSITY (--shape "split chorus:2"). Chorus word density 1.16 -> 2.08
#      words/s on a real take. Across the 23 takes of the batch, chorus/verse
#      measures 1.13 raw, 2.60 with the split alone and 2.51 with hook + split,
#      so the hook costs almost nothing and the two fit together.
#
#   Rejected: --reduce verse:2 (holds each dropped note). Measured, it turned
#   25 s of verse words into sustained vowels and whisper found no words at all.
#
# Same briefs and same --seed-base as the 2026-10-15 run on purpose: the plan is
# seed-determined, so the melodies stay recognisable and the only difference the
# user hears is the density fix -- a batch-level A/B against what they already own.
#
# ASCII only: PowerShell 5.1 reads a BOM-less UTF-8 .ps1 as GBK, and cmd.exe reads
# .bat in the OEM codepage; either one mis-decodes CJK into a command fragment.

$ErrorActionPreference = 'Stop'

$Root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PyYue2 = Join-Path $Root 'vendor\YuE2\.venv\Scripts\python.exe'
$PyPub  = Join-Path $Root 'vendor\ACE-Step-1.5\.venv\Scripts\python.exe'
$Driver = Join-Path $Root 'tools\generation\run_epic_yue2.py'
$Export = Join-Path $Root 'tools\generation\export_delivery.py'
$Review = Join-Path $Root 'tools\generation\battle_review.py'
$LogDir = Join-Path $Root 'logs'

$Batch    = 'outputs\2026-10-16_epic_hook'
$Delivery = 'outputs\2026-10-16_epic_en'
$stamp    = Get-Date -Format 'yyyyMMdd_HHmmss'

Set-Location $Root
Write-Output "[chain] start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 1. Generate. One process per title is the driver's own design (a long-lived
#    process fragments the CUDA allocator and the NAR prefill then OOMs).
& $PyYue2 -I -u $Driver `
    --inputs  'inputs\2026-10-15_epic_en' `
    --out-dir $Batch `
    --hours 22 --candidates 3 --retries 1 `
    --nar-query-chunk 1024 `
    --intro-bars 2 --max-onset 20 `
    --shape 'hook chorus' --shape 'split chorus:2' `
    --min-duration 120 --max-duration 210 `
    --seed-base 1200 --language en
Write-Output "[chain] generator exit=$LASTEXITCODE"

# 2. Dated delivery folder: WAV + same-stem bilingual lyrics TXT + manifest,
#    one folder, per the convention settled 2026-10-13.
& $PyPub -I -u $Export `
    --batch-dir $Batch --engine YuE2 --name epic_en
Write-Output "[chain] export exit=$LASTEXITCODE"

# 3. Chinese listening sheet next to the songs.
$sheet = Join-Path $Delivery '2026-10-16_epic_en_review.md'
& $PyPub -I -u $Review --batch-dir $Batch --engine YuE2 --out $sheet
Write-Output "[chain] review exit=$LASTEXITCODE"

Write-Output "[chain] done $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
