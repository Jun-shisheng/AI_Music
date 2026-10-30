param(
    [string]$Until = "2026-10-22T17:30:00",
    [string]$Log = "D:\Model\AI_Music\logs\yue2_watchdog.log"
)
$target = [datetime]::Parse($Until)
while ((Get-Date) -lt $target) {
    Start-Sleep -Seconds 60
}
$killed = @()
$processes = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -ne $null -and $_.CommandLine -match 'yue2_batch_battle.py'
}
foreach ($process in $processes) {
    try {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction Stop
        $killed += $process.ProcessId
    } catch {
    }
}
$line = "[$(Get-Date -Format 's')] deadline reached; killed: " + ($killed -join ",")
$line | Out-File -FilePath $Log -Append -Encoding utf8
