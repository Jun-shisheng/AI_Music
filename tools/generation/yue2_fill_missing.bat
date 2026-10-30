@echo off
rem Re-run only the YuE2 battle titles that are missing from selected/.
rem
rem Why one process per title: the CUDA OOM that killed 03/04 during the first
rem YuE2 pass is cumulative -- the long-lived process ends up holding fragmented
rem memory and the NAR prefill (which needs ~2.4 GiB in one go) eventually fails.
rem A fresh process per title plus PYTORCH_ALLOC_CONF=expandable_segments:True
rem (the remedy the OOM message itself suggests) plus a smaller NAR query chunk
rem removes both causes.  Already-promoted titles are skipped, so this is safe to
rem run repeatedly.
rem
rem ASCII only: a BOM-less .ps1/.bat with non-ASCII comments gets mis-decoded.
setlocal enabledelayedexpansion

set "PY=D:\Model\AI_Music\vendor\YuE2\.venv\Scripts\python.exe"
set "SCRIPT=D:\Model\AI_Music\tools\generation\yue2_batch_battle.py"
set "SEL=D:\Model\AI_Music\outputs\battle_batch_yue2\selected"

cd /d D:\Model\AI_Music

for %%F in (inputs\battle_batch\*_lyrics.txt inputs\battle_batch2\*_lyrics.txt) do (
    set "NAME=%%~nF"
    set "SLUG=!NAME:_lyrics=!"
    if not exist "%SEL%\!SLUG!.wav" (
        echo [fill] missing: !SLUG!
        set "PYTORCH_ALLOC_CONF=expandable_segments:True"
        "%PY%" -I -u "%SCRIPT%" --inputs inputs\battle_batch --inputs2 inputs\battle_batch2 --out-dir outputs\battle_batch_yue2 --candidates 1 --retries 2 --budget 12 --nar-query-chunk 1024 --backend torch-eager --only !SLUG!
    ) else (
        echo [fill] already present: !SLUG!
    )
)
echo [fill] done
endlocal
