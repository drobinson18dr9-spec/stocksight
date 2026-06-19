# Refresh ALL StockSight forecasts locally with current full-market data, then
# push. Runs the bulk yfinance pull (which only works OFF GitHub Actions, i.e.
# here on your machine / Colab), so every ticker is current to today.
#
# Run manually:  powershell -ExecutionPolicy Bypass -File scripts\refresh_forecasts_local.ps1
# Or install it as a daily task: scripts\install_forecast_refresh_task.ps1
$ErrorActionPreference = "Stop"
$Repo = "C:\Users\drobi\stocksight"
$Py = "C:\Users\drobi\anaconda3\envs\py313\python.exe"
Set-Location $Repo

Write-Host "[$(Get-Date -Format HH:mm)] Pulling full universe via yfinance and computing forecasts..."
& $Py src/build_explorer.py --all-universe --source yf --test-days 12 --horizon 10
if ($LASTEXITCODE -ne 0) { throw "build_explorer failed" }

Write-Host "[$(Get-Date -Format HH:mm)] Committing and pushing fresh forecasts..."
git add -f assets/predict
git commit -q -m "Local full-universe forecast refresh ($(Get-Date -Format yyyy-MM-dd)) [yfinance, current]" 2>$null
# Push via the existing 'origin' remote (uses your saved git credentials / gh auth).
$n = 0
do {
  git push origin HEAD:master
  if ($LASTEXITCODE -eq 0) { break }
  $n++; if ($n -ge 5) { throw "push failed after retries" }
  git fetch origin master; git rebase -X ours origin/master 2>$null; Start-Sleep 4
} while ($true)

Write-Host "[$(Get-Date -Format HH:mm)] Pushed. Triggering Pages redeploy..."
$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (Test-Path $gh) { & $gh workflow run deploy.yml 2>$null }
Write-Host "[$(Get-Date -Format HH:mm)] Done. Site will refresh with current data shortly."
