# Installs the StockSight SMS agent as a hidden, auto-starting Windows task.
# Run ONCE (right-click > Run with PowerShell, or: powershell -ExecutionPolicy Bypass -File this.ps1).
# After this you never open a terminal or start anything again: it launches at
# every login, runs with no visible window, and restarts itself if it crashes.
#
# Uninstall any time:  Unregister-ScheduledTask -TaskName StockSightSMSAgent -Confirm:$false

$ErrorActionPreference = "Stop"
$TaskName = "StockSightSMSAgent"
$Repo     = "C:\Users\drobi\stocksight"
$Script   = Join-Path $Repo "src\sms_agent.py"

# pythonw.exe runs with NO console window (that is what kills the terminal requirement).
$PyDir  = Split-Path "C:\Users\drobi\anaconda3\envs\py313\python.exe"
$Pythonw = Join-Path $PyDir "pythonw.exe"
if (-not (Test-Path $Pythonw)) { $Pythonw = Join-Path $PyDir "python.exe" }  # fallback

if (-not (Test-Path $Script))  { throw "Cannot find $Script" }
if (-not (Test-Path $Pythonw)) { throw "Cannot find python at $Pythonw" }

# Remove any prior copy so re-running this is safe.
try { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop } catch {}

$Action = New-ScheduledTaskAction -Execute $Pythonw -Argument "`"$Script`"" -WorkingDirectory $Repo
$Trigger = New-ScheduledTaskTrigger -AtLogOn
# Hidden, no time limit, auto-restart up to 999 times if it ever exits.
$Settings = New-ScheduledTaskSettingsSet -Hidden -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Settings $Settings -Principal $Principal `
    -Description "StockSight two-way SMS agent (hidden, auto-start at login)." | Out-Null

# Start it right now so you do not have to log out/in to begin.
Start-ScheduledTask -TaskName $TaskName

Write-Host "Installed and started '$TaskName'."
Write-Host "It now runs hidden at every login. No terminal needed."
Write-Host "Stop it:    Stop-ScheduledTask -TaskName $TaskName"
Write-Host "Remove it:  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
Write-Host "Check it:   Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
