# Leo Mission Control — one-click Windows launcher
# Double-click "Leo Mission Control.bat" (or the desktop shortcut) instead of
# running this directly. What it does, silently:
#   1. First run only: creates an SSH key and installs it on the lab box
#      (you type the lab password once — never again after that).
#   2. Makes sure the dashboard server is running on the lab box.
#   3. Opens a background SSH tunnel (no terminal window).
#   4. Opens the dashboard as its own app window (Edge/Chrome), or a browser tab.
# Safe to click any number of times: it never starts duplicates.

param([switch]$Setup)

$ErrorActionPreference = "Stop"
$LabHost   = "irl@10.115.102.210"
$Port      = 8321
$RemoteDir = "~/leorover_work/leorover_isaac"
$Key       = Join-Path $env:USERPROFILE ".ssh\leo_dashboard"

function Show-Msg([string]$text, [string]$title = "Leo Mission Control") {
    try { Add-Type -AssemblyName PresentationFramework
          [System.Windows.MessageBox]::Show($text, $title) | Out-Null }
    catch { Write-Host $text }
}

function Test-Port([int]$p) {
    try { $c = New-Object Net.Sockets.TcpClient
          $ok = $c.ConnectAsync("127.0.0.1", $p).Wait(700); $c.Close(); return $ok }
    catch { return $false }
}

# ---------- 1. one-time key setup ----------
if (-not (Test-Path $Key)) {
    Write-Host ""
    Write-Host "=== First-time setup (only happens once per machine) ==="
    Write-Host "Creating a login key and installing it on the lab box."
    Write-Host "You will be asked for the lab box password ONE time."
    Write-Host "(Make sure the UA VPN is connected first.)"
    Write-Host ""
    New-Item -ItemType Directory -Force -Path (Split-Path $Key) | Out-Null
    & cmd /c "ssh-keygen -t ed25519 -q -N `"`" -f `"$Key`""
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$Key.pub")) {
        Show-Msg "Couldn't create an SSH key. Is Windows OpenSSH installed? (Settings > Optional features > OpenSSH Client)"; exit 1 }
    $pub = Get-Content "$Key.pub" -Raw
    $pub | ssh -o StrictHostKeyChecking=accept-new $LabHost "mkdir -p ~/.ssh && chmod 700 ~/.ssh && tr -d '\r' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    if ($LASTEXITCODE -ne 0) {
        Remove-Item $Key, "$Key.pub" -ErrorAction SilentlyContinue
        Show-Msg "Couldn't reach the lab box to install the key.`n`nCheck that the UA VPN is connected, then double-click again."; exit 1 }
    ssh -i $Key -o BatchMode=yes -o ConnectTimeout=8 $LabHost "true"
    if ($LASTEXITCODE -ne 0) { Show-Msg "Key installed but passwordless login failed — run this again; if it persists, tell Claude."; exit 1 }
    Write-Host "Setup complete. From now on it's a single silent double-click."
    Start-Sleep 1
}

# ---------- 2. make sure the dashboard is running on the lab box ----------
$ensure = "pgrep -f 'dashboard/app.py' >/dev/null || " +
          "(cd $RemoteDir && mkdir -p ~/leo_logs && " +
          "nohup python3 dashboard/app.py > ~/leo_logs/dashboard.log 2>&1 & sleep 1)"
ssh -i $Key -o BatchMode=yes -o ConnectTimeout=8 $LabHost $ensure 2>$null
if ($LASTEXITCODE -ne 0) {
    Show-Msg "Can't reach the lab box (10.115.102.210).`n`nAlmost always this means the UA VPN isn't connected. Connect it and double-click again."; exit 1 }

# ---------- 3. background tunnel (only if not already up) ----------
if (-not (Test-Port $Port)) {
    Start-Process -WindowStyle Hidden ssh -ArgumentList @(
        "-i", $Key, "-N",
        "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        "-L", "${Port}:localhost:${Port}", $LabHost)
    $up = $false
    foreach ($i in 1..24) { Start-Sleep -Milliseconds 350; if (Test-Port $Port) { $up = $true; break } }
    if (-not $up) { Show-Msg "The tunnel didn't come up. Connect the UA VPN and try again."; exit 1 }
}

# ---------- 4. open as an app window ----------
$url = "http://localhost:$Port"
try { Start-Process "msedge.exe" "--app=$url" }
catch { try { Start-Process "chrome.exe" "--app=$url" } catch { Start-Process $url } }
