param(
    [Parameter(Mandatory=$true)][string]$Inputs,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][string]$Language,
    [Parameter(Mandatory=$true)][int]$SeedBase,
    [Parameter(Mandatory=$true)][string]$OutLog,
    [Parameter(Mandatory=$true)][string]$ErrLog,
    [int]$QueryChunk = 1024
)

$root = "D:\Model\AI_Music"
$python = Join-Path $root "vendor\YuE2\.venv\Scripts\python.exe"
$arg = '-u "tools/generation/yue2_batch_battle.py" --inputs "' + $Inputs + '" --out-dir "' + $OutDir + '" ' +
       '--candidates 1 --retries 1 --budget 12 --backend torch-eager --nar-query-chunk ' + $QueryChunk + ' ' +
       '--language ' + $Language + ' --seed-base ' + $SeedBase + ' --intro-bars 2 ' +
       '--shape "split chorus:2" --shape "thin verse:1.5" --final-lift 2 --early-accept'
$process = Start-Process -FilePath $python -ArgumentList $arg -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru
Write-Output $process.Id
