# connect-runpod.ps1 - one-shortcut session starter for RunPod + VS Code
#
# What it does (in order):
#   1. Checks pod status via RunPod API
#   2. If pod is STOPPED, sends start command and polls until RUNNING
#   3. Updates ~/.ssh/config with current pod IP/port
#   4. Launches VS Code Remote-SSH at /workspace/IsaacLab (with -LaunchVSCode)
#
# Usage:
#   .\connect-runpod.ps1                    # config + maybe start, no VS Code
#   .\connect-runpod.ps1 -LaunchVSCode      # full session start (recommended)
#   .\connect-runpod.ps1 -NoStart           # only update config; don't start a stopped pod
#   .\connect-runpod.ps1 -SshHost my-other-pod # different ~/.ssh/config Host alias (default: runpod)
#
# Setup (one-time):
#   1. Generate API key at https://www.runpod.io/console/user/settings
#      -> "API Keys" tab -> "Create API Key" -> copy the key
#   2. Find your pod ID in the dashboard URL when viewing the pod:
#      https://www.runpod.io/console/pods/abc123xyz789  <- "abc123xyz789" is the ID
#   3. Save both as User-level env vars (persist across PowerShell sessions):
#         [System.Environment]::SetEnvironmentVariable('RUNPOD_API_KEY', 'rpa_xxxxx', 'User')
#         [System.Environment]::SetEnvironmentVariable('RUNPOD_POD_ID', 'abc123xyz789', 'User')
#      Close and reopen PowerShell after running these.

[CmdletBinding()]
param(
    [switch]$LaunchVSCode,
    [switch]$NoStart,
    [string]$SshHost = "runpod",
    [string]$RemoteFolder = "/workspace/IsaacLab",
    [int]$StartTimeoutSec = 300
)

$ErrorActionPreference = "Stop"

# --- Validate config ------------------------------------------------------
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

# --- Fetch pod info from RunPod REST API ---------------------------------
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

# --- Check pod status; start it if needed --------------------------------
$status = $pod.desiredStatus

if ($status -ne "RUNNING") {
    if ($NoStart) {
        Write-Warning "Pod status is '$status' and -NoStart was passed. Exiting."
        exit 1
    }

    Write-Host "[runpod] Pod is '$status', sending start command..." -ForegroundColor Yellow

    # Helper to surface the actual response body from RunPod, not just HTTP status
    function Get-ApiErrorBody {
        param($ErrorRecord)
        try {
            $stream = $ErrorRecord.Exception.Response.GetResponseStream()
            $reader = New-Object System.IO.StreamReader($stream)
            return $reader.ReadToEnd()
        } catch {
            return ""
        }
    }

    # Try several known endpoints + body shapes + methods. RunPod's API surface
    # has changed over versions; we attempt each until one works. Modern (2025+)
    # API uses PATCH with desiredStatus, older versions use POST /start or /resume.
    $startAttempts = @(
        @{ method = "PATCH"; url = "https://rest.runpod.io/v1/pods/$PodId"; body = '{"desiredStatus":"RUNNING"}'; desc = "PATCH / with desiredStatus=RUNNING" },
        @{ method = "POST";  url = "https://rest.runpod.io/v1/pods/$PodId/start";  body = "{}"; desc = "POST /start with empty body" },
        @{ method = "POST";  url = "https://rest.runpod.io/v1/pods/$PodId/start";  body = '{"gpuCount":1}'; desc = "POST /start with gpuCount=1" },
        @{ method = "POST";  url = "https://rest.runpod.io/v1/pods/$PodId/resume"; body = "{}"; desc = "POST /resume with empty body" }
    )

    $startSucceeded = $false
    foreach ($attempt in $startAttempts) {
        Write-Host "[runpod]   trying: $($attempt.desc)" -ForegroundColor DarkGray
        try {
            Invoke-RestMethod -Uri $attempt.url -Headers $headers -Method $attempt.method `
                -Body $attempt.body -TimeoutSec 30 | Out-Null
            $startSucceeded = $true
            Write-Host "[runpod]   success on: $($attempt.desc)" -ForegroundColor Green
            break
        }
        catch {
            $apiBody = Get-ApiErrorBody $_
            Write-Host "[runpod]     failed: $($_.Exception.Message)" -ForegroundColor DarkGray
            if ($apiBody) { Write-Host "[runpod]     response: $apiBody" -ForegroundColor DarkGray }
        }
    }

    if (-not $startSucceeded) {
        Write-Error @"
All start attempts failed. RunPod's API rejected each known endpoint+body shape.

Most likely causes:
- API key lacks write permissions. Check at https://www.runpod.io/console/user/settings - your API key needs 'Read & Write' access, not just 'Read'.
- Pod was terminated (not just stopped) and no longer exists. Check the dashboard.
- Insufficient credit balance - top up via dashboard.
- Required GPU type is unavailable in the region right now.

Fallback: start the pod manually in the dashboard, then re-run this script with -NoStart to just update the SSH config:
    .\connect-runpod.ps1 -LaunchVSCode -NoStart
"@
        exit 1
    }

    # Poll until RUNNING (or timeout)
    Write-Host "[runpod] Waiting for pod to boot (timeout: ${StartTimeoutSec}s)..." -ForegroundColor Cyan
    $deadline = (Get-Date).AddSeconds($StartTimeoutSec)
    $lastStatus = ""

    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try {
            $pod = Invoke-RestMethod -Uri "https://rest.runpod.io/v1/pods/$PodId" `
                -Headers $headers -Method GET -TimeoutSec 15
            $status = $pod.desiredStatus
            if ($status -ne $lastStatus) {
                Write-Host "[runpod]   status: $status" -ForegroundColor DarkGray
                $lastStatus = $status
            }
            if ($status -eq "RUNNING") { break }
        }
        catch {
            Write-Host "[runpod]   transient API error, retrying..." -ForegroundColor DarkGray
        }
    }

    if ($status -ne "RUNNING") {
        Write-Error "Pod did not reach RUNNING within ${StartTimeoutSec}s. Check dashboard: https://www.runpod.io/console/pods/$PodId"
        exit 1
    }

    Write-Host "[runpod] Pod is RUNNING." -ForegroundColor Green
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

# --- Extract SSH port mapping --------------------------------------------
# RunPod's API response shape for portMappings:
#   [ { "privatePort": 22, "publicPort": 22119, "type": "tcp", "ip": "69.30.85.76" }, ... ]
# Some accounts/responses use 'runtime.ports' instead - handle both.
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

# --- Update ~/.ssh/config ------------------------------------------------
$configPath = "$env:USERPROFILE\.ssh\config"
if (-not (Test-Path $configPath)) {
    "" | Out-File -FilePath $configPath -Encoding ascii -NoNewline
}

$existing = Get-Content $configPath -Raw

$newEntry = @"
Host $SshHost
    HostName $publicIp
    Port $publicPort
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
    ServerAliveInterval 60
    ServerAliveCountMax 3
"@

# Replace any existing entry for this Host alias, or append if missing.
$pattern = "(?ms)^Host\s+$([regex]::Escape($SshHost))\s*$.*?(?=^Host\s|\z)"
if ($existing -match $pattern) {
    $updated = [regex]::Replace($existing, $pattern, "$newEntry`r`n")
} else {
    $updated = $existing.TrimEnd() + "`r`n`r`n$newEntry`r`n"
}

$updated | Out-File -FilePath $configPath -Encoding ascii -NoNewline
Write-Host "[runpod] Updated $configPath - 'ssh $SshHost' now points at $publicIp`:$publicPort" -ForegroundColor Green

# --- Optional: launch VS Code Remote-SSH ---------------------------------
if ($LaunchVSCode) {
    Write-Host "[runpod] Launching VS Code Remote-SSH at $RemoteFolder..." -ForegroundColor Cyan
    try {
        & code --remote "ssh-remote+$SshHost" $RemoteFolder
        Write-Host "[runpod] VS Code launched. First connection may take ~30 sec to install remote server." -ForegroundColor Cyan
    }
    catch {
        Write-Warning "Failed to launch VS Code: $($_.Exception.Message). Open it manually and connect via F1 -> Remote-SSH: Connect to Host -> $SshHost"
    }
}
