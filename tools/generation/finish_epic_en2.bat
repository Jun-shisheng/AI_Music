@echo off
rem Run the 2026-10-15 English epic batch, then build the dated delivery folder
rem and push it to the private repo -- the same chain the user asked for on
rem 10-14 ("如果关机前完成了就自动上传没有就算了").
rem
rem This batch is the answer to the user's three notes on the 10-14 delivery:
rem   * no long intro  -> --intro-bars 2 (plan edit) + --max-onset 20 (gate)
rem   * bigger range   -> new caption + lyrics, shape-aware candidate ranking
rem   * less repetition-> one hook, twice, in the middle; new intro/final/outro
rem
rem ASCII only -- cmd.exe reads .bat in the OEM codepage.
setlocal

cd /d D:\Model\AI_Music
set "PY_YUE2=D:\Model\AI_Music\vendor\YuE2\.venv\Scripts\python.exe"
set "PY_PUB=D:\Model\AI_Music\vendor\ACE-Step-1.5\.venv\Scripts\python.exe"

echo [chain] start %DATE% %TIME%

"%PY_YUE2%" -I -u tools\generation\run_epic_yue2.py ^
  --inputs inputs\2026-10-15_epic_en ^
  --out-dir outputs\2026-10-15_epic_en ^
  --hours 6 --candidates 2 --retries 1 ^
  --nar-query-chunk 1024 --intro-bars 2 --max-onset 20 ^
  --min-duration 120 --max-duration 210
echo [chain] generator exit=%ERRORLEVEL%

"%PY_PUB%" -I -u tools\generation\publish_delivery.py ^
  --batch-dir outputs\2026-10-15_epic_en --engine YuE2 --name epic_en
echo [chain] publish exit=%ERRORLEVEL%

echo [chain] done %DATE% %TIME%
endlocal
