# scripts/ — Standalone helpers

| script | runs on | purpose |
|--------|---------|---------|
| `connect-runpod.ps1` | Windows / PowerShell | Auto-update `~/.ssh/config` from RunPod API + optionally launch VS Code Remote-SSH |
| `pod_setup.sh` | Linux / on the pod | Boot script: restore SSH host keys, authorized_keys, bashrc from `/workspace` on every container start |
| `train.py` | Linux / on the pod | Phase 2 stub: wrapper around Isaac Lab's `rsl_rl/train.py` with leorover defaults |
| `eval.py` | Linux / on the pod | Phase 2 stub: load a checkpoint, run N episodes, output CSV matching PyBullet schema |
| `compare_hybrid_vs_lqr.py` | Linux / on the pod | Phase 7 stub: port of `run_comparison` mode |
| `convert_assets.sh` | Linux / on the pod | Phase 1 stub: one-liner for URDF → USD conversion |

The training-related scripts are stubs filled in at their port phase. The
helpers (`connect-runpod.ps1` and `pod_setup.sh`) are usable today.

---

## `connect-runpod.ps1` — Windows-side SSH config auto-updater

Eliminates the manual "edit notepad with new IP/port" step every time
your pod restarts. Queries RunPod's REST API, parses the current SSH
port mapping, rewrites your `~/.ssh/config` `Host runpod` entry, and
optionally launches VS Code Remote-SSH targeting the pod.

### One-time setup

1. **Generate a RunPod API key.** Go to
   https://www.runpod.io/console/user/settings → **API Keys** tab →
   **Create API Key** → name it `ssh-config-script` → copy the key
   (starts with `rpa_`).

2. **Find your pod ID.** In the RunPod dashboard, click on your pod.
   The URL bar will show something like
   `https://www.runpod.io/console/pods/abc123xyz789` — the ID is the
   last path segment (`abc123xyz789`).

3. **Save both as User-level Windows env vars.** In PowerShell:

   ```powershell
   [System.Environment]::SetEnvironmentVariable('RUNPOD_API_KEY', 'rpa_xxxxxxxxxxxxxxxxxxxx', 'User')
   [System.Environment]::SetEnvironmentVariable('RUNPOD_POD_ID',  'abc123xyz789',             'User')
   ```

   **Close that PowerShell window and open a fresh one** — env vars set
   this way only show up in new sessions.

4. **Verify they're set in the fresh window:**

   ```powershell
   echo $env:RUNPOD_API_KEY
   echo $env:RUNPOD_POD_ID
   ```

### Per-session usage

After clicking Start in the RunPod dashboard and waiting ~90 seconds for
the pod to boot:

```powershell
cd C:\Users\Aaron\Downloads\leorover_isaac\scripts
.\connect-runpod.ps1 -LaunchVSCode
```

That single command:

- Queries RunPod's API for current pod IP/port
- Updates `~/.ssh/config`'s `Host runpod` entry
- Launches VS Code with Remote-SSH already targeting the pod
- Drops you into `/workspace/IsaacLab`

Skip `-LaunchVSCode` if you only want the config updated (e.g., to use
`ssh runpod` from PowerShell first for `git pull` or similar).

### Troubleshooting

- **"RUNPOD_API_KEY env var not set"** — env var was set but not in
  this PowerShell window. Close and reopen.
- **"Pod status is 'STOPPED'"** — start the pod from the dashboard
  first, then re-run.
- **"Port 22 is not exposed"** — edit the pod's template, add `22` to
  "Expose TCP Ports", save. Most templates expose 22 by default.
- **"RunPod API call failed: ... 401"** — API key is invalid or
  revoked. Generate a new one, update the env var, reopen PowerShell.
- **VS Code opens but Remote-SSH fails** — the `code` CLI isn't on
  your PATH. Open VS Code manually → F1 → **Shell Command: Install
  'code' command in PATH** → close VS Code → retry script.

### What this script does NOT do

- **Start or stop the pod.** Manual via dashboard. If you ever want
  to add that, RunPod's API has `/v1/pods/{id}/start` and `/v1/pods/{id}/stop`
  endpoints — small extension.
- **Sync files.** Use `git` for code, `scp`/`rsync` for one-off data.
- **Monitor costs.** Check the RunPod dashboard's Billing page.

---

## `pod_setup.sh` — pod-side boot script

Restores SSH host keys, `authorized_keys`, and bashrc tweaks from the
persistent `/workspace` volume on every container start. Without this,
every Stop/Start cycle wipes those files (they live in the container
layer, not the volume).

See the file's header comment for the install procedure, and CLOUD_SETUP.md
in the repo root for the full per-pod walkthrough.

---

## Phase-stubs (Linux / on the pod)

`train.py`, `eval.py`, `compare_hybrid_vs_lqr.py`, `convert_assets.sh`
are placeholder files for their respective port phases. They raise
`NotImplementedError` until their phase is built out — see
PORTING_ROADMAP.md for the phase plan.
