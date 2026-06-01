# stop-runpod.ps1 - stop a running RunPod pod to halt GPU billing
#
# Usage:
#   .\stop-runpod.ps1                # stops the pod identified by RUNPOD_POD_ID (prompts to confirm)
#   .\stop-runpod.ps1 -PodId xyz789  # override pod ID
#   .\stop-runpod.ps1 -Force         # skip the confirmation prompt (use in desktop shortcut)
#
# What it does:
#   1. Checks pod status via RunPod API
#   2. If RUNNING, sends stop command (which halts GPU billing immediately)
#   3. Persistent volume keeps your /workspace data intact
#
# What it does NOT do:
#   - Terminate the pod (deletes everything, can't undo)
#   - Sync any files first - make sure you've committed/pushed any code
#     changes before running this

[CmdletBinding()]
param(
    [string]$PodId = $env:RUNPOD_POD_ID,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ApiKey = $env:RUNPOD_API_KEY
if (-not $ApiKey) {
    Write-Error "RUNPOD_API_KEY env var not set. See scripts/README.md for setup."
    exit 1
}
if (-not $PodId) {
    Write-Error "RUNPOD_POD_ID env var not set (or pass -PodId). See scripts/README.md."
    exit 1
}

$headers = @{
    'Authorization' = "Bearer $ApiKey"
    'Content-Type'  = 'application/json'
}

# Get current status
Write-Host "[runpod] Checking pod $PodId..." -ForegroundColor Cyan
try {
    $pod = Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$PodId" `
        -Headers $headers -Method GET -TimeoutSec 15
}
catch {
    Write-Error "Failed to query pod status: $($_.Exception.Message)"
    exit 1
}

$status = $pod.desiredStatus
Write-Host "[runpod] Current status: $status" -ForegroundColor Cyan

if ($status -eq "STOPPED" -or $status -eq "EXITED") {
    Write-Host "[runpod] Pod is already stopped. Nothing to do." -ForegroundColor Green
    exit 0
}

if ($status -ne "RUNNING") {
    Write-Warning "Pod is in transitional state '$status'. Stop command may fail. Try again in a moment if it does."
}

# Confirmation prompt (skip with -Force)
if (-not $Force) {
    $reply = Read-Host "Stop pod $PodId? GPU billing halts immediately. (y/N)"
    if ($reply -notmatch '^[yY]') {
        Write-Host "[runpod] Cancelled." -ForegroundColor Yellow
        exit 0
    }
}

# Send stop
Write-Host "[runpod] Sending stop command..." -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$PodId/stop" `
        -Headers $headers -Method POST -TimeoutSec 30 | Out-Null
}
catch {
    Write-Error @"
Failed to stop pod via API: $($_.Exception.Message)

Common causes:
- API key lacks write permissions
- RunPod API outage (rare; check https://status.runpod.io)

If the script fails, just stop the pod manually from the dashboard:
https://www.runpod.io/console/pods/$PodId
"@
    exit 1
}

Write-Host "[runpod] Stop command sent. Pod will shut down within ~30 seconds." -ForegroundColor Green
Write-Host "[runpod] Verify in dashboard: https://www.runpod.io/console/pods/$PodId" -ForegroundColor Cyan
