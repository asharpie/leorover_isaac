# Leo Rover (Isaac Lab) — Runbook

Everything you need to **train the rover, watch it learn, and look at results**, on the lab
RTX 4090. If you can copy-paste, you can run this project.

It all goes through one helper command called **`leo`**. You almost never type the long
`run_lab.sh ... train.py ...` commands by hand anymore — `leo` does it for you.

---

## TL;DR — the whole workflow

```bash
leo train hybrid     # start a full training run (4096 rovers, the good reward settings, in the background)
leo watch            # live view of it booting + learning   (Ctrl-C to stop watching, NOT the run)
leo curve            # is it learning? success% + reward over time
leo trace            # evaluate the newest checkpoint: how many rovers reach the goal
leo stop             # end the run cleanly when you're done
```

That's 90% of day-to-day use. The rest of this doc explains each piece, how to read the
numbers, and what to do when something looks off.

---

## 1. One-time setup (≈2 minutes)

**a. Get on the machine.** Connect to the UA VPN, then SSH in:

```bash
ssh irl@10.115.102.210
```

(The password lives in your own notes — don't put it in any file in this repo.)

**b. Go to the project and get the latest code:**

```bash
cd ~/leorover_work/leorover_isaac
git pull
```

**c. Install the `leo` shortcut (once):**

```bash
chmod +x scripts/leo.sh scripts/run_lab.sh
echo "alias leo='$HOME/leorover_work/leorover_isaac/scripts/leo.sh'" >> ~/.bashrc
source ~/.bashrc
```

Now you can type `leo ...` from anywhere. (If you skip the alias, just run
`scripts/leo.sh ...` from inside the repo folder instead.)

Type `leo help` any time to see the menu.

---

## 2. Train

Start the main, validated hybrid run:

```bash
leo train hybrid
```

This launches **4096 rovers in parallel** on the Mars terrain with the curriculum and the
**fixed reward settings** that actually learn, fully headless, **in the background** (it keeps
running after you log out), and writes a log under `~/leo_logs/`.

Other options:

| Command | What it does |
|---|---|
| `leo train ppo` | Train the **pure-PPO** policy (no LQR helper) — the comparison baseline |
| `leo train hybrid --envs 2048` | Use fewer parallel rovers (less GPU memory) |
| `leo train hybrid --iters 2000` | Stop automatically after 2000 iterations |
| `leo train hybrid --raw` | Use the **stock** reward weights (the ones that *don't* learn) — for A/B only |
| `leo train hybrid --fg` | Run in the foreground (you watch it live; closing the terminal kills it) |

When it starts it prints the log path and reminds you of `leo watch` / `leo curve` / `leo stop`.
It also shows current GPU usage first — if a labmate's job is already on the card, **wait or
coordinate, don't kill their work** (see Troubleshooting).

---

## 3. Watch it learn

```bash
leo watch     # live tail of the run. Ctrl-C just stops watching; the run keeps going.
```

You'll see a block every iteration. The lines that matter:

- **`Mean reward`** — was about **−200** when the policy was broken. **Rising toward 0 and
  positive = it's learning.**
- **`Mean action noise std`** — how much the policy is still exploring. **~0.2 to 0.7 is
  healthy.** If it crashes to **~0.02**, exploration collapsed (the old failure).
- **`Mean episode length`** — how long rovers survive before finishing/failing. Higher =
  they're driving, not stalling out.
- **`success=NN%`** lines (from the curriculum) — the fraction of rovers reaching the goal.
  **This climbing is the headline signal.**

For a quick scan of just those trends without the firehose:

```bash
leo curve
```

It prints the recent success%, reward, and std so you can see the direction at a glance.

> **Note:** `Mean reward` and `success%` only appear once some episodes have *finished*.
> Episodes are long (~1,000–2,000 steps) and each iteration only advances 32 steps, so for the
> first ~40 iterations those lines are blank. That's normal, not a bug.

---

## 4. Stop a run

```bash
leo stop
```

It lists *your* training processes, asks for confirmation, then force-stops them. **Always use
this instead of `kill`** — the Isaac simulator ignores a normal `kill`, and a half-killed run
keeps hogging ~11 GB of the GPU and silently halves everyone's speed. `leo stop` handles that
correctly and only ever touches your own processes.

---

## 5. Look at results

### How far does a trained policy get? (`leo trace`)

```bash
leo trace            # newest hybrid checkpoint
leo trace 600        # a specific one: model_600.pt
```

This runs the saved policy on a thousand rovers and prints a **POPULATION** summary like:

```
========== POPULATION over all 1024 envs (best progress reached) ==========
  ever reached goal    : 712 / 1024  (69.5%)
  best path_progress   : mean 71.3%  median 88.0%  p90 99.0%  max 100.0%
  envs >25% progress   : 901 / 1024
  envs <5%  progress   : 41 / 1024
```

**Read the POPULATION block, not the single "env 0" summary above it** — one rover can be lucky
or unlucky; the population is the real measure. It also saves a top-down picture of the path at
`logs/leo_rover_mars_hybrid/<run>/eval_trace/trace.png`, and prints the exact `scp` line to copy
that image to your laptop.

### List what's been saved

```bash
leo checkpoints      # the model_*.pt files of the latest run (saved every 200 iterations)
```

### Full training curves in TensorBoard

```bash
leo tb               # starts TensorBoard on the lab box and prints the tunnel command
```

Then, in a **separate terminal on your laptop**, open the tunnel it tells you to:

```bash
ssh -L 6006:localhost:6006 irl@10.115.102.210
```

and open **http://localhost:6006** in your browser. You get reward, episode length, std, and the
losses plotted over the whole run. (Ctrl-C in the `leo tb` window stops TensorBoard.)

---

## 6. What "good" looks like (reading the numbers)

| Signal | Where | Broken | Healthy |
|---|---|---|---|
| `success=` % | `leo curve` / `leo watch` | stuck at 0% | climbing toward 70–100% |
| Mean reward | `leo curve` / `leo watch` | flat near −200 | rising toward 0 and positive |
| Mean action noise std | `leo curve` / `leo watch` | crashes to ~0.02 early | holds ~0.2–0.7, drifts down late |
| `ever reached goal` | `leo trace` | 0 / N | a large fraction of N |
| `path_progress` | `leo trace` | ~4% (parked at the start) | median well above 50% |

Rough rule: if **success% is climbing** and **reward is rising** while **std hasn't collapsed**,
the run is healthy — let it cook. Checkpoints are saved every 200 iterations, so you never lose
progress.

---

## 7. The reward knobs (only if you want to tune)

`leo train` already sets the **validated** values automatically. The fix that made it learn:

| Setting | Stock | Fixed | Why |
|---|---|---|---|
| `LEOROVER_W_EFFORT` | 0.5 | **0.05** | Stock value punished the policy so hard for using its controls that "do nothing" was optimal. |
| `LEOROVER_W_PROGRESS` | 10 | **150** | Make driving forward the dominant reward (the rover crawls slowly, so progress per step is small). |
| `LEOROVER_W_SMOOTH` | 0.5 | **0.1** | Stop over-penalising the control changes needed to steer. |
| `LEOROVER_ENT_COEF` | 0.001 | **0.005** | Keep the policy exploring so it doesn't freeze before it learns to drive. |

Other tunable knobs (defaults shown): `LEOROVER_W_CTE` (5), `LEOROVER_W_VELOCITY` (0.5),
`LEOROVER_W_HEADING` (0.5), `LEOROVER_NUM_STEPS` (32, rollout length per update).

To **experiment** with a different value, use the raw command (Appendix A) with your own value
in front, e.g. test a stronger progress weight:

```bash
LEOROVER_W_PROGRESS=200 scripts/run_lab.sh scripts/train.py \
  --task Isaac-LeoRover-Mars-Hybrid-v0 --num_envs 4096 --headless
```

> Tip for the final, cleanest policy: once success% is consistently high, lower
> `LEOROVER_ENT_COEF` toward `0.002` so the policy stops exploring and sharpens.

---

## 8. Troubleshooting

**The GPU is busy / I share the box.** Run `leo gpu`. If a process using more than ~2–3 GB is
*not* yours (e.g. a labmate's `aurora` job), **wait or ask them — never kill someone else's
process.** `leo stop` only ever stops your own runs. Two big runs on one card just makes both
crawl.

**`git pull` says "Already up to date" but I changed a file on Windows.** Editing on Windows
isn't enough — you must `git add`, `git commit`, and `git push` from Windows **first**, then
`git pull` on the lab box. The lab box only sees what's been pushed.

**There's no live 3D window / video.** Correct — the graphics renderer crashes on this machine's
GPU driver, so everything runs **headless** on purpose. To "watch" an episode, use `leo trace`,
which produces a top-down plot of the path instead of a live view.

**A run died after ~25 seconds during loading.** It was probably interrupted, or the load just
hadn't finished. Loading takes ~60–90 seconds (terrain build + simulator start) before training
actually begins. Just relaunch and let it run.

**`leo` says `bad interpreter` or shows `^M` errors.** The script picked up Windows line endings.
Fix once on the box: `sed -i 's/\r$//' scripts/leo.sh`.

**`leo: command not found`.** The alias didn't load. Re-run the `echo ... >> ~/.bashrc` +
`source ~/.bashrc` from setup, or just call `scripts/leo.sh` from inside the repo.

**Did my run actually use the fix?** During startup it prints `[reward override] ...` lines.
Check the log: `grep "reward override" ~/leo_logs/*.log` — you should see `ppo_w_effort = 0.05`,
`ppo_w_progress = 150.0`, `ppo_w_smoothness = 0.1`.

---

## Appendix A — the raw commands (what `leo` runs for you)

You normally don't need these, but for transparency:

```bash
# Train (what `leo train hybrid` runs):
LEOROVER_W_EFFORT=0.05 LEOROVER_W_PROGRESS=150 LEOROVER_W_SMOOTH=0.1 LEOROVER_ENT_COEF=0.005 \
nohup scripts/run_lab.sh scripts/train.py \
  --task Isaac-LeoRover-Mars-Hybrid-v0 --num_envs 4096 --headless \
  > ~/leo_logs/hybrid_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Evaluate a checkpoint at scale (what `leo trace` runs):
scripts/run_lab.sh scripts/trace_episode.py --task Isaac-LeoRover-Mars-Hybrid-v0 \
  --checkpoint logs/leo_rover_mars_hybrid/<run>/model_<N>.pt --num_envs 1024 --steps 1500

# Stop everything of yours (what `leo stop` runs):
pkill -9 -u "$USER" -f 'scripts/train.py'
```

## Appendix B — where things live

| Path | What |
|---|---|
| `~/leorover_work/leorover_isaac` | the project (on the lab box) |
| `scripts/leo.sh` | the `leo` helper |
| `scripts/run_lab.sh` | low-level Isaac launcher (sets display/CUDA env) |
| `scripts/train.py`, `scripts/trace_episode.py` | training + evaluation entry points |
| `logs/<experiment>/<timestamp>/` | one folder per run |
| &nbsp;&nbsp;`model_*.pt` | checkpoints (every 200 iterations) |
| &nbsp;&nbsp;`csv/episode_metrics.csv` | per-episode stats (success, path_progress, reward, …) |
| &nbsp;&nbsp;`eval_trace/trace.png` | top-down plot from `leo trace` |
| `~/leo_logs/*.log` | the console log of each run you started with `leo` |

Experiment folder names: hybrid → `leo_rover_mars_hybrid`, pure-PPO → `leo_rover_mars`,
flat → `leo_rover_flat`.

## Appendix C — task names

| `leo` alias | Gym task id | Description |
|---|---|---|
| `hybrid` | `Isaac-LeoRover-Mars-Hybrid-v0` | LQR baseline + PPO residual on Mars terrain (main) |
| `ppo` | `Isaac-LeoRover-Mars-v0` | pure PPO on Mars terrain (comparison) |
| `flat` | `Isaac-LeoRover-Flat-v0` | flat-ground smoke test |
