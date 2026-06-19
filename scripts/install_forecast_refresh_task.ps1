# Installs a daily Windows task that refreshes ALL StockSight forecasts with
# current full-market data (yfinance, which works here but not on GitHub Actions)
# and pushes them. Runs at 6:30am local. Run ONCE.
#
# Uninstall: Unregister-ScheduledTask -TaskName StockSightForecastRefresh -Confirm:$false
$ErrorActionPreference = "Stop"
$TaskName = "StockSightForecastRefresh"
$Script = "C:\Users\drobi\stocksight\scripts\refresh_forecasts_local.ps1"
if (-not (Test-Path $Script)) { throw "Cannot find $Script" }

try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop } catch {}

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Script`""
$Trigger = New-ScheduledTaskTrigger -Daily -At 6:30am
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Settings $Settings -Principal $Principal `
    -Description "Daily local full-universe forecast refresh (yfinance, current data)." | Out-Null

Write-Host "Installed '$TaskName' (runs daily 6:30am)."
Write-Host "Run it now once to seed:  powershell -ExecutionPolicy Bypass -File $Script"
Write-Host "Remove it:  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
