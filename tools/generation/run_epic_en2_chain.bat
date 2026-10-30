@echo off
rem Corrected ASCII-only launcher for the 2026-10-15 English epic batch.
rem
rem Why this file exists next to finish_epic_en2.bat: that one carried the same
rem chain but with Chinese text inside its `rem` comments, and cmd.exe reads
rem .bat in the OEM codepage. The UTF-8 bytes of the Chinese were decoded as
rem GBK, a byte landed on a command separator, and cmd tried to execute the
rem fragments ('ish', 'ate' and the tail of a comment) as commands. Harmless
rem this time -- the generator line survived and the run started -- but it is
rem exactly the trap the repo notes warn about. Keep .bat files ASCII only.
rem
rem Chain: 6 h of generation, then the dated delivery folder + git push.
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
