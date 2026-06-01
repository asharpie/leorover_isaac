# Cloud GPU Setup — RunPod + Isaac Lab

End-to-end guide for running Isaac Lab on a rented cloud GPU from your
Windows desktop, keeping your local machine free for everything else
(including gaming during training runs).

This is the **recommended path** for this project. The dual-boot
alternative is in [INSTALL.md](INSTALL.md) if you ever want a fully local
setup. See [SETUP_HISTORY.md](SETUP_HISTORY.md) for why we landed here
rather than WSL2.

---

## Why RunPod specifically

We picked RunPod over the alternatives (Lambda Labs, Vast.ai, AWS) for
three reasons:

1. **Community templates with Isaac Sim pre-installed.** RunPod's
   marketplace has user-shared environments where Isaac Sim 4.5 and Isaac
   Lab are already configured. Saves you the 1–2 hour first-time install
   we kept failing at in WSL2.
2. **Per-minute billing.** Spin up for a 20-minute experiment and pay for
   20 minutes, not a full hour.
3. **Predictable pricing.** Unlike Vast.ai's auction marketplace, RunPod
   pricing is fixed per GPU type so you know what a session will cost
   before you start.

If RunPod doesn't work out for you, the same flow applies to Lambda Labs
and Vast.ai with minor naming differences.

---

## What you'll need on the Windows side

- [ ] **Windows Terminal** (free from Microsoft Store) — much nicer SSH
      experience than legacy PowerShell. Optional but recommended.
- [ ] **Built-in OpenSSH client** — comes with Windows 10/11 by default,
      no install needed. Verify: open PowerShell and run `ssh -V`. Should
      print something like `OpenSSH_for_Windows_8.6p1`.
- [ ] **A code editor** for editing files locally before syncing — VS Code
      is the obvious pick. https://code.visualstudio.com/
- [ ] **Credit card** for RunPod billing (no upfront commitment, pay as
      you go).

---

## Total expected setup time

- Phase 1 (account + SSH key): 15 minutes
- Phase 2 (first pod launch + connect): 10 minutes
- Phase 3 (smoke test): 10 minutes
- Phase 4 (clone repo + leorover_isaac install): 10 minutes

You can have your first Isaac Lab training run on a cloud GPU within an
hour of starting this guide.

---

## Phase 1 — RunPod account and SSH key

### 1.1 Sign up at runpod.io

Go to https://runpod.io and create an account. They support GitHub login,
which is convenient if you already have one.

Add a payment method under **Billing → Add Funds**. You can pre-load credit
(e.g., $20) rather than auto-pay. I'd suggest starting with a $10 or $20
preload — RunPod won't auto-launch anything, you have to manually start
pods, so there's no risk of unexpected charges as long as you actually
shut down what you start.

### 1.2 Generate an SSH key pair on Windows

SSH uses public-key cryptography. You generate a pair of files: a *private*
key you keep on your laptop, a *public* key you give to RunPod. When you
connect, the two are matched cryptographically.

Open PowerShell and run:

```powershell
ssh-keygen -t ed25519 -C "your-email@example.com"
```

When it asks for a file location, just press Enter to accept the default
(`C:\Users\Aaron\.ssh\id_ed25519`). When it asks for a passphrase, you
can either set one (more secure, you type it each time you connect) or
leave it empty (less secure but easier — fine for a personal cloud GPU
account).

This created two files:
- `C:\Users\Aaron\.ssh\id_ed25519` — your **private** key. Never share.
- `C:\Users\Aaron\.ssh\id_ed25519.pub` — your **public** key. This is
  the one you give to RunPod.

### 1.3 Upload the public key to RunPod

Display the public key:

```powershell
cat $env:USERPROFILE\.ssh\id_ed25519.pub
```

Copy the entire output (starts with `ssh-ed25519 AAAA...`, ends with your
email).

In RunPod: click your avatar in the top right → **Settings** → **SSH
Public Keys** → **Add Public Key**. Paste it in, give it a name like
"aaron-laptop", save.

This key now lets you SSH into any pod you launch without a password.

---

## Phase 2 — Launch your first Isaac Lab pod

### 2.1 Pick a GPU type

In RunPod, click **GPU Pods** → **Deploy**. You'll see a list of GPU types
with hourly prices. For Isaac Lab:

| GPU | VRAM | typical cost | when to pick |
|-----|------|--------------|--------------|
| RTX 4090 | 24 GB | ~$0.40–0.60/hr | **Default choice.** Best price/performance for Isaac Lab. |
| RTX A5000 | 24 GB | ~$0.30–0.40/hr | Cheaper alternative if 4090 is unavailable. |
| RTX A6000 | 48 GB | ~$0.70–0.90/hr | If you hit VRAM limits with many parallel envs. |
| A100 | 40/80 GB | $1.30–2.50/hr | Overkill for now. Pick for very long final training runs. |

Start with **RTX 4090**. It's enough for thousands of parallel envs on
the leorover_isaac scaffold.

### 2.2 Pick a template

Look for templates with "Isaac" or "Isaac Lab" in the name. Search the
template browser; community templates are tagged.

If none are current (templates come and go), use the official
**runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04** template as
a base — it's clean PyTorch + CUDA on Ubuntu 22.04, and you can install
Isaac Sim on top yourself. I'll cover that case below.

### 2.3 Configure storage

- **Container Disk:** 30 GB — for the OS and quick scratch space. Wiped
  when the pod terminates.
- **Volume Disk:** 50 GB, mount at `/workspace` — **persistent storage**
  that survives pod restarts. Your code, models, and training data go
  here. Pay a small fee (~$0.02/GB/month) even when the pod is off.

The persistent volume is the key cost-saving trick: you can shut down the
pod entirely (paying $0/hour for GPU), and your work-in-progress stays
safe on the volume. Next time you launch, attach the same volume and
everything is exactly where you left it.

### 2.4 Optional but recommended settings

- **Idle Timeout:** set to 30 minutes. If you forget about a pod, it
  auto-shuts-down after 30 min of zero GPU utilization. Major cost
  savings if you ever forget.
- **Expose HTTP Ports:** leave blank for now (you'd add 6006 here later
  if you wanted to view TensorBoard from your browser).

### 2.5 Click Deploy

Pod boots in 1–3 minutes. You'll see it transition through "Provisioning"
→ "Starting" → "Running."

---

## Phase 3 — SSH in and verify the environment

### 3.1 Get the SSH command

In the RunPod dashboard, click your running pod → **Connect** → **SSH
over exposed TCP**. You'll see a command like:

```
ssh root@123.45.67.89 -p 12345 -i ~/.ssh/id_ed25519
```

**IMPORTANT — IP and port change every time you Stop/Start a pod.** If you
stop a pod and start it again later, you must go back to the dashboard
and copy the new SSH command. The old one will fail with `Connection
refused`. Your data on the persistent volume survives the restart, but
the pod's network identity is fresh each time. This is one of the more
common surprises for new RunPod users.

Copy that.

### 3.2 Connect from PowerShell

Paste the command into PowerShell. First time will prompt:

```
The authenticity of host '[123.45.67.89]:12345' can't be established.
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type `yes` and press Enter. You're now on the remote pod.

Your prompt will change to something like `root@a1b2c3d4e5f6:/workspace#`.
You're in.

### 3.3 Smoke-test the GPU

```bash
nvidia-smi
```

Should show your rented GPU (probably an RTX 4090).

### 3.4 Smoke-test Isaac Sim

If the template included Isaac Sim:

```bash
which python && python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('Isaac Sim OK'); app.close()"
```

You should see Isaac Sim's startup wall of warnings ending with
`Isaac Sim OK`. First run takes a few minutes (shader cache).

If the template *didn't* include Isaac Sim, install it now (one-time per
pod template, then save your own template — see Phase 5):

```bash
pip install --upgrade pip
pip install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
```

Then smoke-test again.

### 3.5 Install Isaac Lab

```bash
cd /workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

(If the template included this, it'll detect existing install and skip.)

Verify with the bundled cartpole task:

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 \
    --num_envs 64 \
    --headless \
    --max_iterations 20
```

Iteration logs should scroll with rewards climbing. You've now done what
WSL2 wouldn't let you do.

---

## Phase 4 — Clone leorover_isaac and start porting

### 4.1 Clone your repo

```bash
cd /workspace
git clone https://github.com/asharpie/leorover_isaac.git
cd leorover_isaac
pip install -e .
```

Verify:

```bash
python -c "import leorover_isaac; print(leorover_isaac.__file__)"
```

### 4.2 Set up Git credentials on the pod

You'll want to commit and push from the pod, not just from your laptop.
Configure git identity:

```bash
git config --global user.name "Sharpie"
git config --global user.email "aaronsharp2005@gmail.com"
```

For pushing to GitHub, use a personal access token (https://github.com/settings/tokens)
the first time you push; git will cache it.

---

## Phase 5 — The cost-saving workflow

This is the most important phase. Cloud bills get scary when people leave
pods running. Here's the discipline that keeps costs reasonable.

### 5.1 When you're done for the session

**Don't just close the SSH window** — that leaves the pod running and
billing you. From the RunPod dashboard:

- **Stop the pod**: stops the GPU billing immediately. Container disk is
  wiped (including any pip-installed packages!), but `/workspace` volume
  disk persists (charges ~$2-3/month). Next launch: pod boots in 1-2
  minutes with files on `/workspace` intact but a fresh container layer.
- **Terminate the pod**: deletes everything including container. Cheapest
  but you lose any uncommitted state outside `/workspace`. Volume disk
  is separate and still persists.

Use **Stop**, not **Terminate**, for routine sessions. Use Terminate
only when you're done with a project entirely.

**CRITICAL — Stop wipes the container layer, including Python packages.**
When you Stop a pod, anything installed via `pip install` or `apt install`
on the container's root filesystem (the default install location) gets
wiped. Only `/workspace` survives. So after a Stop/Start cycle, you'll
hit errors like `No such file or directory: '_isaac_sim/python.sh'` —
that's the install being gone, not your data.

**The fix is to immediately save a custom template after your first
successful install** (see Section 5.2 below). With a saved template, the
Isaac Sim install is baked into the container image, and future pods boot
ready-to-train.

### 5.2 Save a custom template once Isaac Sim works

After your first successful Isaac Sim install, save the pod's state as a
custom template so future launches skip the install entirely:

1. In RunPod dashboard → your pod → **More** → **Save as Template**
2. Give it a name like "isaac-lab-leorover"
3. Future deployments: pick this template, your Isaac Sim install is
   baked in. Pod boots ready-to-train in 2 minutes.

This is the single biggest workflow improvement — once it's saved, every
future session is "click Deploy, wait 2 min, SSH in, train."

### 5.3 Set a monthly budget alarm

RunPod → **Billing** → **Auto Pay Settings** → set a low monthly limit
(e.g., $50). RunPod won't charge past that without your action. Acts as
a safety net for forgotten pods.

### 5.4 Routine cost-control habits

- Always **Stop** the pod when stepping away for more than 30 min
- Enable the **30-min idle auto-shutdown** at pod launch
- Check the **Billing** page weekly to spot any creep
- Never launch a pod and then walk away without setting either Stop or
  idle-timeout — this is how people get a $200 surprise bill

### 5.5 Typical session cost

For a 2-hour Isaac training session on an RTX 4090 at $0.50/hr:

- GPU time: 2 hours × $0.50 = $1.00
- Volume disk (persistent): ~$0.05 for a day's worth of storage
- **Total per session: ~$1.05**

Over a month of part-time research (8 sessions × ~2 hours): ~$10/month.

---

## Phase 6 — Transferring data between local and pod

### 6.1 Copy a file FROM the pod TO Windows

In a PowerShell terminal (not SSH'd in):

```powershell
scp -i $env:USERPROFILE\.ssh\id_ed25519 -P 12345 root@123.45.67.89:/workspace/leorover_isaac/logs/episode_metrics.csv C:\Users\Aaron\Downloads\
```

Substitute your pod's port and IP from the RunPod Connect screen.

### 6.2 Copy a file FROM Windows TO the pod

```powershell
scp -i $env:USERPROFILE\.ssh\id_ed25519 -P 12345 C:\Users\Aaron\Downloads\some_data.npy root@123.45.67.89:/workspace/leorover_isaac/data/
```

### 6.3 Push code changes via git (preferred)

For source code, don't copy files directly. Edit locally on Windows
(using VS Code or whatever), commit, push to GitHub, then on the pod:

```bash
cd /workspace/leorover_isaac
git pull
```

This keeps everything version-controlled and avoids the "which version
is on which machine" confusion.

### 6.4 VS Code Remote-SSH (the slick way)

If you install the **Remote - SSH** extension in VS Code, you can open
files directly on the pod from your local VS Code window. Edits save
to the pod over SSH; it feels exactly like editing local files but the
files live remotely.

1. VS Code → Extensions → install "Remote - SSH" (Microsoft)
2. F1 → "Remote-SSH: Connect to Host..." → paste `ssh root@<ip> -p <port>`
3. VS Code reconnects with the pod as its filesystem
4. Open `/workspace/leorover_isaac` like any local folder

This is the workflow most people end up preferring — gives you the
"feels local" editing experience while the code stays on the pod.

---

## Phase 7 — Watching long training runs

You can close your SSH session and the training keeps running on the pod.
Two ways to monitor:

### 7.1 Reattach later via SSH

Reconnect and use `tail -f` on your training log:

```bash
tail -f /workspace/leorover_isaac/logs/latest_run/train.log
```

### 7.2 TensorBoard or wandb in your browser

If you want graphical training curves while gaming, set up TensorBoard:

1. At pod launch, expose port **6006** (Pod settings → Expose HTTP Ports)
2. On the pod: `tensorboard --logdir /workspace/leorover_isaac/logs --bind_all`
3. RunPod gives you a URL like `https://abc-6006.proxy.runpod.net/`
4. Open in your browser — TensorBoard runs while you play Rocket League
   on the same screen

`wandb` works the same way without needing port exposure (it pushes to
wandb.ai's servers which you view from any browser).

---

## Phase 8 — `tmux` for training that survives disconnects

If your internet hiccups, an SSH disconnect kills any program running in
that session. Solve this by running training inside `tmux`, a session
manager that keeps processes alive across disconnects.

```bash
# Start a named tmux session
tmux new -s training

# Inside tmux, run your training command
./isaaclab.sh -p ~/IsaacLab/source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-LeoRover-Mars-v0 --num_envs 1024 --headless

# Detach (training keeps running): Ctrl+B then D

# Disconnect SSH entirely if you want

# Later, reconnect via SSH, then reattach to tmux:
tmux attach -t training
```

This is the standard ML researcher pattern for long training runs. Learn
the two commands above and you're set.

---

## Troubleshooting

### Pod won't start, says "no capacity"
RunPod is out of the GPU type you asked for in your region. Try a
different region (click the region selector at pod launch) or a different
GPU (RTX A5000 if 4090 is full).

### SSH says "Permission denied (publickey)"
Either your private key isn't where SSH expects it, or the public key
you uploaded to RunPod doesn't match. Check:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -v root@<ip> -p <port>
```

The `-v` shows verbose debug output. Most common fix: re-copy your public
key from `cat $env:USERPROFILE\.ssh\id_ed25519.pub` and re-paste to
RunPod settings.

### Isaac Sim crashes with the same Vulkan error from WSL2
Almost impossible on RunPod (their hosts are real NVIDIA hardware), but if
it happens: terminate the pod and launch a new one. Could be a one-off
host issue.

### "I forgot to shut down a pod and got a big bill"
Email RunPod support — they sometimes credit accidental overcharges,
especially if you can show you stopped the pod immediately upon noticing.
Set the idle-timeout next time.

### Pod runs out of disk during training
Default container disk (30 GB) is tight when Isaac caches shaders and
your logs accumulate. Either:
- Increase Container Disk at launch (go to 50 GB)
- Save big outputs (models, logs) to `/workspace` (the persistent volume)
- Stream wandb logs externally instead of writing to disk

### Connection feels slow / laggy
SSH terminal latency is ~50–150 ms depending on your distance to the pod
region. Pick a region close to you (US-East from Florida; US-West from
California; EU regions if in Europe).

### How do I delete an SSH key from Windows
```powershell
Remove-Item $env:USERPROFILE\.ssh\id_ed25519
Remove-Item $env:USERPROFILE\.ssh\id_ed25519.pub
```
Then remove it from RunPod's settings too.

---

## Where to go next

1. [PORTING_ROADMAP.md](PORTING_ROADMAP.md) — the actual port plan, phase by phase
2. [leorover_isaac/envs/README.md](leorover_isaac/envs/README.md) — PyBullet → Isaac Lab env mapping
3. [SETUP_HISTORY.md](SETUP_HISTORY.md) — the WSL2/Docker debugging story and why cloud was chosen
4. [INSTALL.md](INSTALL.md) — dual-boot Ubuntu fallback if you ever want a purely local setup

When you're done for the day: **stop the pod**. When you start again
tomorrow: **start the pod, SSH in, you're back where you left off**.
The whole point of this setup is that it stays out of your way until
you actively want to use it.
