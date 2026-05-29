# Installing Isaac Sim + Isaac Lab on WSL2

End-to-end install guide for setting up Isaac Lab on Windows 11 via WSL2
Ubuntu 22.04, targeting an NVIDIA RTX 4060 Ti.

If anything in this guide drifts out of date (Isaac releases roughly every
3–6 months), the authoritative source is always
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html.

---

## What you're installing, and why

You will end up with three layers stacked on top of each other:

1. **WSL2 + Ubuntu 22.04** — Linux running inside Windows, sharing the NVIDIA
   GPU through the Windows driver. Isaac Lab is much better supported on Linux
   than native Windows; WSL2 gives you Linux without giving up your Windows
   desktop.
2. **Isaac Sim** — NVIDIA's robotics simulator (built on Omniverse). Provides
   the underlying GPU physics engine (PhysX), USD asset pipeline, and
   rendering.
3. **Isaac Lab** — RL training framework that sits on top of Isaac Sim. Replaces
   the deprecated Isaac Gym. Provides task/environment APIs, vectorized
   observations, integration with PPO/SAC implementations (rsl_rl, skrl,
   stable-baselines3).

Total disk footprint after install is roughly **40–60 GB**.

---

## Prerequisites checklist

Before starting, verify each of these on your Windows host:

- [ ] **Windows 11** (or Windows 10 build 19044+). Check with `winver`.
- [ ] **NVIDIA RTX 4060 Ti** visible in Device Manager.
- [ ] **NVIDIA Game Ready or Studio driver ≥ 545.84**. Check at
      https://www.nvidia.com/Download/index.aspx. The Windows driver is
      sufficient — you do NOT install a separate Linux driver inside WSL2.
- [ ] **At least 60 GB free** on your Windows drive (WSL2 lives on the C: drive
      by default unless you move it).
- [ ] **32 GB system RAM** strongly recommended. 16 GB will work for small
      tasks but you'll hit OOM on anything with >256 parallel envs.

---

## Step 1 — Install WSL2 + Ubuntu 22.04

Open **PowerShell as Administrator** on Windows:

```powershell
wsl --install -d Ubuntu-22.04
```

This installs the WSL2 kernel and Ubuntu 22.04 in one shot. Reboot when
prompted, then launch Ubuntu from the Start menu. On first launch it asks
you to set a Linux username and password — pick something memorable, this
is a per-distro account.

Verify after reboot:

```powershell
wsl --status
wsl -l -v
```

You should see `Ubuntu-22.04` running on **VERSION 2**. If it shows VERSION 1,
fix it:

```powershell
wsl --set-version Ubuntu-22.04 2
```

---

## Step 2 — Verify GPU access from inside WSL2

Open the Ubuntu shell (`wsl` from any terminal, or the Ubuntu Start menu
entry) and run:

```bash
nvidia-smi
```

You should see your RTX 4060 Ti listed with the same driver version Windows
reports. **If `nvidia-smi` fails or shows "command not found", stop and fix
this before proceeding** — every Isaac install step below depends on it.

Common fix: update your Windows NVIDIA driver. The driver in WSL2 is forwarded
from Windows; there is no separate Linux driver to install.

---

## Step 3 — Install system dependencies inside WSL2

In the Ubuntu shell:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libegl1 \
    libxkbcommon0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1
```

The libGL/libX libraries are needed by Isaac Sim's renderer even in headless
mode (it still initializes the graphics stack).

---

## Step 4 — Install Miniconda

Isaac Lab strongly prefers conda over a bare venv because of how it manages
the Isaac Sim Python distribution.

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
exec bash
```

Confirm:

```bash
conda --version
which conda
```

---

## Step 5 — Install Isaac Sim (pip method)

As of Isaac Sim 4.5+, NVIDIA supports a pure-pip install that pulls Isaac
Sim from PyPI rather than the heavy Omniverse Launcher route. This is much
simpler and what Isaac Lab's docs now recommend.

```bash
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# Pin CUDA wheel exactly — Isaac Lab requires PyTorch 2.4.0 with CUDA 11.8
# at the time of writing. Check the Isaac Lab release notes for the version
# matching your Isaac Sim release.
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# Isaac Sim from NVIDIA's PyPI index
pip install --upgrade pip
pip install isaacsim==4.5.0.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-extscache-physics==4.5.0.0 \
            isaacsim-extscache-kit==4.5.0.0 \
            isaacsim-extscache-kit-sdk==4.5.0.0 \
            --extra-index-url https://pypi.nvidia.com
```

The version pins above are for Isaac Sim 4.5. Newer versions usually have
matching pin sets documented in the Isaac Lab release page — substitute as
needed.

Smoke-test that Isaac Sim launches in headless mode:

```bash
python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('Isaac Sim OK'); app.close()"
```

First launch takes 2–5 minutes (it compiles shader caches). Subsequent
launches are 10–30 seconds. If you see GPU initialization output and "Isaac
Sim OK", you're good.

---

## Step 6 — Install Isaac Lab

Clone the Isaac Lab repo into your home directory and run its installer:

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

The installer installs the Isaac Lab Python package and its RL backends
(rsl_rl, skrl) into the active conda env. It takes 5–15 minutes.

---

## Step 7 — Smoke-test the install

Run one of the bundled demo tasks to confirm GPU physics + RL pipeline both
work:

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/standalone/demo/play_camera.py --headless
```

If that runs without errors, try a tiny RL run on a built-in task:

```bash
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 \
    --num_envs 64 \
    --headless \
    --max_iterations 20
```

You should see iteration logs scrolling with reward numbers climbing. This
confirms: GPU is being used, the RL loop runs end-to-end, and your install
is healthy.

---

## Step 8 — Install leorover_isaac as an editable package

Now wire up the project that lives next to Isaac Lab:

```bash
# The Windows path C:\Users\Aaron\Downloads\... is reachable from WSL via /mnt/c
cd "/mnt/c/Users/Aaron/Downloads"
# Move the leorover_isaac/ folder so it's a true sibling of the PyBullet repo
# (if you haven't already)
ls   # should show both leoroverpybullet_share... and leorover_isaac
cd leorover_isaac
pip install -e .
```

The `-e` flag makes it editable, so changes to source files take effect
without reinstalling. Verify:

```bash
python -c "import leorover_isaac; print(leorover_isaac.__file__)"
```

---

## Step 9 — Set up your shell aliases (optional but worth it)

Add to `~/.bashrc`:

```bash
# Auto-activate Isaac Lab env in new shells
alias isaac='conda activate isaaclab && cd ~/IsaacLab'
alias lab='~/IsaacLab/isaaclab.sh -p'
alias leorov='cd "/mnt/c/Users/Aaron/Downloads/leorover_isaac"'

# Headless training shortcut — most of your runs will be like this
alias train_headless='~/IsaacLab/isaaclab.sh -p ~/IsaacLab/source/standalone/workflows/rsl_rl/train.py --headless'
```

`source ~/.bashrc` to apply.

---

## Step 10 — Initialize the git repository

The scaffolding shipped without a `.git` directory so you can choose where
to push. Inside WSL:

```bash
cd "/mnt/c/Users/Aaron/Downloads/leorover_isaac"
git init
git lfs install            # for large USD assets later
git add .
git commit -m "Initial scaffolding: WSL2 install + Isaac Lab port skeleton"
# Create the GitHub repo (gh CLI is convenient; otherwise create via web)
gh auth login              # if first time
gh repo create leorover_isaac --public --source=. --remote=origin --push
```

---

## Troubleshooting

### `nvidia-smi: command not found` inside WSL2
- Make sure you're on **Windows 11** or **Windows 10 21H2+**. Older Windows 10 doesn't support GPU-in-WSL2.
- Update the Windows NVIDIA driver to ≥545.84.
- After driver update, run `wsl --shutdown` from Windows PowerShell, then relaunch the Ubuntu shell.

### Isaac Sim crashes with "Failed to create swapchain" or "Vulkan error"
- WSL2's GPU forwarding sometimes doesn't expose a display surface. Force
  headless mode: `SimulationApp({'headless': True})`. All training should be
  headless anyway.

### Conda env is named `isaaclab` but Python is wrong version
- The `python=3.10` pin in `conda create` matters. Isaac Sim 4.5 only supports
  3.10. Recreate the env if you ended up with 3.11 or 3.12.

### "CUDA out of memory" with even small env counts
- The 4060 Ti (8 GB variant) is tight. Start with `--num_envs 64`, not 4096.
  Wheeled-robot tasks with terrain heightfields are heavier than the
  cartpole-style examples.
- Close any browser/game open on the Windows side — they share VRAM with WSL2.

### Slow file I/O when assets live on the Windows side
- WSL2's `/mnt/c/...` access is slow for many small files (USD asset trees can
  be thousands of files). For real work, copy assets into the WSL2 filesystem
  (`/home/<you>/leorover_isaac/...`) and either symlink back to Windows or work
  entirely Linux-side. For first scaffolding, working from `/mnt/c/` is fine.

### `./isaaclab.sh --install` fails with "permission denied"
- `chmod +x isaaclab.sh` inside the IsaacLab repo, then retry.

---

## Where to go next

1. Read `PORTING_ROADMAP.md` for the phase-by-phase port plan.
2. Read `leorover_isaac/envs/README.md` for the PyBullet → Isaac Lab env
   mapping in detail.
3. The Isaac Lab tutorials are excellent starting points:
   - https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html
4. Pay particular attention to the "Creating a Direct Workflow RL Env"
   tutorial — that's the API your Leo Rover task will use.
