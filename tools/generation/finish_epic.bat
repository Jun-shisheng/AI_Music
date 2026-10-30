@echo off
rem Top up the 2026-10-14 epic-English batch (titles 10-12) and, as soon as the
rem generation stops for any reason, build the delivery folder and push it to
rem the private repo. The user asked for exactly this: upload if it finished
rem before shutdown, otherwise leave it.
rem
rem Only titles 10-12 are touched: a resumed run re-checks every promoted take
rem against the duration gate, and the ceiling was tightened to 210 s mid-run, so
rem without --slug-filter this would silently re-roll 02/04/06 (~1.5 h).
rem
rem ASCII only -- cmd.exe reads .bat in the OEM codepage.
setlocal

cd /d D:\Model\AI_Music
set "PY_YUE2=D:\Model\AI_Music\vendor\YuE2\.venv\Scripts\python.exe"
set "PY_PUB=D:\Model\AI_Music\vendor\ACE-Step-1.5\.venv\Scripts\python.exe"

echo [chain] start %DATE% %TIME%

"%PY_YUE2%" -I -u tools\generation\run_epic_yue2.py ^
  --inputs inputs\2026-10-14_epic_en ^
  --out-dir outputs\2026-10-14_epic_en ^
  --slug-filter 10_ 11_ 12_ ^
  --hours 2.5 --candidates 2 --retries 1 ^
  --nar-query-chunk 1024 --min-duration 120 --max-duration 210 ^
  --seed-base 1300
echo [chain] generator exit=%ERRORLEVEL%

"%PY_PUB%" -I -u tools\generation\publish_delivery.py ^
  --batch-dir outputs\2026-10-14_epic_en --engine YuE2
echo [chain] publish exit=%ERRORLEVEL%

echo [chain] done %DATE% %TIME%
endlocal
