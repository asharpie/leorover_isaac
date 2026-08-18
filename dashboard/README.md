# Leo Mission Control — the project dashboard

One web page for the whole workflow: launch training/evals, watch runs live,
browse every checkpoint and data file, build analysis figures, and export
paper-ready PNG/SVG charts. Zero dependencies — pure Python 3 stdlib
(pandas is used automatically if present, but is not required).

## Run it

On the lab box (recommended — that's where the data lives):

    cd ~/leorover_work/leorover_isaac
    git pull
    python3 dashboard/app.py

Then from your laptop or PC:

    ssh -L 8321:localhost:8321 irl@10.115.102.210

and open http://localhost:8321 in any browser. (Same trick as TensorBoard.)

You can also run it directly on Windows against local data copies:

    py dashboard\app.py --repo C:\path\to\leorover_isaac

On a machine without a GPU it switches to view-only: every Launch form still
builds the exact command for you to copy into a lab-box terminal.

## What's inside

- **Overview** — GPU, running jobs (with safe stop), newest runs, recent logs.
- **Launch** — train / paired eval / multi-world / quick eval / trace /
  diagnose / record / advanced, with the validated reward fix as defaults and
  a live command preview. Uses the same `scripts/leo.sh` commands as the CLI.
- **Live training** — success%, mean reward, action-std charts parsed from the
  run log, ADR deterministic-eval markers, health badges, console tail.
- **Runs & checkpoints** — every run under `logs/`, all `model_*.pt`,
  diagnostic images, one click into analysis.
- **Training analysis** — the standard research figures for any
  episode_metrics CSV (learning curves, terrain/friction breakdowns,
  residual usage, distributions) plus the `leo_report` text report.
  Corrupt rows (steps > 2000) are excluded by default.
- **Evaluations** — paired suites (success/CTE by level, by geometry,
  per-scenario ΔCTE histogram, paired t-test + McNemar via
  `scripts/paired_stats.py`), multi-world forest plot, quick-eval comparison,
  collapse curves, and the 3-D demo replays.
- **Data explorer** — plot any column vs any column of any CSV (binned line,
  scatter, histogram, grouped bars) with filters; paged raw rows. Handles
  millions of rows server-side.
- **Files** — browse the repo and `~/leo_logs`, preview text/images, download.
- **Paper figures** toggle (top bar) — white/serif chart styling and 4x-res
  PNG/SVG export for the paper.

Port/host options: `--port 8321 --host 127.0.0.1` (default binds localhost
only; use the SSH tunnel rather than exposing it).

## One-click launcher for Windows (laptop & PC)

No terminals needed. In `dashboard/windows/`:

1. Double-click **`Create Desktop Shortcut.bat`** once — puts a
   "Leo Mission Control" icon on your Desktop.
2. Double-click the icon. The **first** run asks for the lab-box password
   one time (it installs a login key); every run after that is silent:
   it makes sure the dashboard is running on the lab box, opens a hidden
   SSH tunnel, and pops the dashboard in its own app window.

Requirements: UA VPN connected, and Windows' built-in OpenSSH client
(present by default on Windows 10/11). Clicking it again when everything
is already running just reopens the window — it never starts duplicates.

## Watching episodes in 3D

Mission Control -> "Watch episodes". Recordings are ground truth: the recorder
samples the running simulation (pose 5x/s, wheel angles, terrain raycast,
actual waypoints), and the viewer renders the real Leo Rover model (converted
from the original ROS meshes) on the real terrain with slip-coloured trails.
Record from the same page (GPU must be free), then "Watch" opens the viewer:
drag to orbit, follow/chase cameras, scrub, speed control. Hybrid (red flag)
and pure LQR (blue flag) replay the same scenario simultaneously; they overlap
until their behaviour diverges - that divergence is the point.
