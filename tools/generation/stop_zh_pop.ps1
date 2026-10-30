$ErrorActionPreference = "SilentlyContinue"
$procs = Get-CimInstance Win32_Process | Where-Object {
    ($_.CommandLine -like "*run_zh_pop_20261026*") -or
    ($_.CommandLine -like "*yue2_batch_battle*")
}
foreach ($p in $procs) {
    Write-Output ("kill " + $p.ProcessId + " " + $p.Name)
    Stop-Process -Id $p.ProcessId -Force
}
