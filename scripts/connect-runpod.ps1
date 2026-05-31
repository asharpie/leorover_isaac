# connect-runpod.ps1 — auto-update ~/.ssh/config with current RunPod IP/port
#
# Usage:
#   .\connect-runpod.ps1                    # just update the SSH config
#   .\connect-runpod.ps1 -LaunchVSCode      # also launch VS Code Remote-SSH
#   .\connect-runpod.ps1 -Host my-other-pod # use a different ~/.ssh/config Host name (default: runpod)
#
# Setup (one-time):
#   1. Generate API key at https://www.runpod.io/console/user/settings
#      → "API Keys" tab → "Create API Key" → copy the key
#   2. Find your pod ID in the dashboard URL when viewing the pod:
#      https://www.runpod.io/console/pods/abc123xyz789  ← "abc123xyz789" is the ID
#   3. Save both as User-level env vars (persist across PowerShell sessions):
#         [System.Environment]::SetEnvironmentVariable('RUNPOD_API_KEY', 'rpa_xxxxx', 'User')
#         [System.Environment]::SetEnvironmentVariable('RUNPOD_POD_ID', 'abc123xyz789', 'User')
#      Close and reopen PowerShell after running these.
#
# Then any time the pod restarts and gets a new IP/port, just run:
#   .\connect-runpod.ps1 -LaunchVSCode
# and you're connected.

[CmdletBinding()]
param(
    [switch]$LaunchVSCode,
    [string]$Host = "runpod",
    [string]$RemoteFolder = "/workspace/IsaacLab"
)

$ErrorActionPreference = "Stop"

# ─── Validate config ──────────────────────────────────────────────────────
$ApiKey = $env:RUNPOD_API_KEY
$PodId = $env:RUNPOD_POD_ID

if (-not $ApiKey) {
    Write-Error @"
RUNPOD_API_KEY env var not set.

Generate one at: https://www.runpod.io/console/user/settings
Then run (and reopen PowerShell):
  [System.Environment]::SetEnvironmentVariable('RUNPOD_API_KEY', 'rpa_xxxxx', 'User')
"@
    exit 1
}

if (-not $PodId) {
    Write-Error @"
RUNPOD_POD_ID env var not set.

Find your pod ID in the dashboard URL (e.g., '.../console/pods/abc123xyz789').
Then run (and reopen PowerShell):
  [System.Environment]::SetEnvironmentVariable('RUNPOD_POD_ID', 'abc123xyz789', 'User')
"@
    exit 1
}

# ─── Fetch pod info from RunPod REST API ─────────────────────────────────
Write-Host "[runpod] Querying pod $PodId..." -ForegroundColor Cyan

$headers = @{
    'Authorization' = "Bearer $ApiKey"
    'Content-Type'  = 'application/json'
}

try {
    $pod = Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$PodId" `
        -Headers $headers -Method GET -TimeoutSec 15
}
catch {
    Write-Error @"
RunPod API call failed: $($_.Exception.Message)

Common causes:
- Invalid API key (check RUNPOD_API_KEY)
- Wrong pod ID (check RUNPOD_POD_ID)
- Pod was terminated and no longer exists
- RunPod API outage (rare; check https://status.runpod.io)
"@
    exit 1
}

# ─── Check pod status ────────────────────────────────────────────────────
$status = $pod.desiredStatus
if ($status -ne "RUNNING") {
    Write-Warning "Pod status is '$status', not RUNNING. Start it in the dashboard first."
    Write-Host "Dashboard: https://www.runpod.io/console/pods/$PodId" -ForegroundColor Yellow
    exit 1
}

# Wait a moment if the runtime block isn't populated yet (pod just started)
$tries = 0
while (-not $pod.portMappings -and $tries -lt 6) {
    Write-Host "[runpod] Pod is RUNNING but port mappings not yet exposed. Waiting 5s..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    $pod = Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$PodId" `
        -Headers $headers -Method GET -TimeoutSec 15
    $tries++
}

# ─── Extract SSH port mapping ────────────────────────────────────────────
# RunPod's API response shape for portMappings:
#   [ { "privatePort": 22, "publicPort": 22119, "type": "tcp", "ip": "69.30.85.76" }, ... ]
# Some accounts/responses use 'runtime.ports' instead — handle both.
$portMappings = $null
if ($pod.portMappings) {
    $portMappings = $pod.portMappings
} elseif ($pod.runtime -and $pod.runtime.ports) {
    $portMappings = $pod.runtime.ports
}

if (-not $portMappings) {
    Write-Error @"
No port mappings found in pod info. Raw response:
$($pod | ConvertTo-Json -Depth 10)

This usually means the pod is still booting. Wait 30 seconds and re-run.
"@
    exit 1
}

$sshMapping = $portMappings | Where-Object {
    ($_.privatePort -eq 22) -or ($_.containerPort -eq 22) -or ($_.privatePort -eq "22")
} | Select-Object -First 1

if (-not $sshMapping) {
    Write-Error @"
Port 22 is not exposed on this pod. Available ports:
$($portMappings | ConvertTo-Json -Depth 5)

Edit the pod and add 22 to 'Expose TCP Ports', then save and retry.
"@
    exit 1
}

$publicIp = $sshMapping.ip
if (-not $publicIp) { $publicIp = $sshMapping.publicIp }
$publicPort = $sshMapping.publicPort

if (-not $publicIp -or -not $publicPort) {
    Write-Error @"
Could not parse IP/port from mapping:
$($sshMapping | ConvertTo-Json -Depth 5)
"@
    exit 1
}

Write-Host "[runpod] Pod is live at $publicIp`:$publicPort" -ForegroundColor Green

# ─── Update ~/.ssh/config ────────────────────────────────────────────────
$configPath = "$env:USERPROFILE\.ssh\config"
if (-not (Test-Path $configPath)) {
    "" | Out-File -FilePath $configPath -Encoding ascii -NoNewline
}

$existing = Get-Content $configPath -Raw

$newEntry = @"
Host $Host
    HostName $publicIp
    Port $publicPort
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
    ServerAliveInterval 60
    ServerAliveCountMax 3
"@

# Replace any existing entry for this Host alias, or append if missing.
$pattern = "(?ms)^Host\s+$([regex]::Escape($Host))\s*$.*?(?=^Host\s|\z)"
if ($existing -match $pattern) {
    $updated = [regex]::Replace($existing, $pattern, "$newEntry`r`n")
} else {
    $updated = $existing.TrimEnd() + "`r`n`r`n$newEntry`r`n"
}

$updated | Out-File -FilePath $configPath -Encoding ascii -NoNewline
Write-Host "[runpod] Updated $configPath — 'ssh $Host' now points at $publicIp`:$publicPort" -ForegroundColor Green

# ─── Optional: launch VS Code Remote-SSH ─────────────────────────────────
if ($LaunchVSCode) {
    Write-Host "[runpod] Launching VS Code Remote-SSH at $RemoteFolder..." -ForegroundColor Cyan
    try {
        & code --remote "ssh-remote+$Host" $RemoteFolder
        Write-Host "[runpod] VS Code launched. First connection may take ~30 sec to install remote server." -ForegroundColor Cyan
    }
    catch {
        Write-Warning "Failed to launch VS Code: $($_.Exception.Message). Open it manually and connect via F1 → Remote-SSH: Connect to Host → $Host"
    }
}
