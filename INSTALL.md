# Installing Ubuntu 22.04 (dual-boot) for Isaac Lab

End-to-end install guide for setting up Isaac Sim + Isaac Lab on a fresh
Ubuntu 22.04 LTS install, dual-booted alongside Windows on an NVIDIA RTX 4060
Ti machine.

If you're wondering why we're not using WSL2: we tried, and it doesn't work
on Sharpie's current Windows build (Insider 26200.x). See
[SETUP_HISTORY.md](SETUP_HISTORY.md) for the full debugging story. Short
version: NVIDIA's WSL graphics passthrough doesn't expose the Vulkan/OpenGL
libraries on this combination of Windows + driver, and Isaac Sim won't run
without them. Native Linux bypasses the entire problem.

If anything in this guide drifts out of date, the authoritative sources are:

- Ubuntu install: https://ubuntu.com/tutorials/install-ubuntu-desktop
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- NVIDIA driver on Ubuntu: https://ubuntu.com/server/docs/nvidia-drivers-installation

---

## Total expected effort

- Phase 1–3 (Windows prep + Ubuntu install): ~2 hours
- Phase 4 (NVIDIA driver + CUDA): ~30 minutes
- Phase 5 (Isaac Sim + Isaac Lab): ~30 minutes
- Phase 6 (verify with leorover_isaac): ~15 minutes

Plan for a half-day. Don't squeeze it into a lunch break.

---

## What you'll need before starting

- [ ] **Free disk space:** at least **250 GB** unused on your Windows drive.
      Isaac Sim's installs and caches plus Ubuntu base plus your training
      runs add up. 300+ is more comfortable.
- [ ] **USB drive:** 8 GB or larger, contents will be erased.
- [ ] **A second device** to read this guide from while you're rebooting
      (phone, tablet, second laptop).
- [ ] **Windows recovery backup** to a USB or external drive — see Phase 1.
- [ ] **An external backup of anything irreplaceable.** Dual-boot installs
      are very safe, but disk partitioning carries some risk. If the worst
      happens, you want to be able to reinstall Windows without losing data.

---

## Phase 1 — Back up Windows and disable conflicting features

Skipping any of these steps risks data loss or boot problems. Do all four
before touching the disk.

### 1.1 Back up your data

Copy anything irreplaceable to an external drive or cloud storage. Your
URCA paper materials, the PyBullet repo, photos, anything else you'd be
upset to lose. Dual-boot setup is statistically very safe but partitioning
is the one step where things can go wrong; budget 30 minutes for backup.

### 1.2 Disable BitLocker (if enabled)

BitLocker encrypts your Windows partition and will interfere with shrinking
or reading the partition during install. Open PowerShell as Administrator:

```powershell
Get-BitLockerVolume
```

If any of your drives show `ProtectionStatus: On`, decrypt them:

```powershell
Disable-BitLocker -MountPoint "C:"
```

This can take minutes to hours depending on drive size — let it finish before
proceeding. Re-encrypt after dual-boot is working if you want, but don't
encrypt the Linux partition.

### 1.3 Disable Fast Startup

Fast Startup keeps Windows in a partial hibernation state on "shutdown,"
which can leave the NTFS filesystem in a state Ubuntu can't mount safely.

1. Open **Control Panel** (not Settings) → **Power Options** → **Choose what the power buttons do**
2. Click **Change settings that are currently unavailable**
3. Uncheck **Turn on fast startup (recommended)**
4. Save changes

### 1.4 Create a Windows recovery USB

Just in case. Search "Create a recovery drive" in Start, follow the wizard.
Needs a separate 16+ GB USB drive. If you skip this and the install
corrupts something, recovery becomes much harder.

---

## Phase 2 — Shrink Windows partition

You're going to carve out unallocated space on your drive for Ubuntu to
install into. Ubuntu will partition that space itself during install.

1. Open **Disk Management** (right-click Start → Disk Management).
2. Find your Windows drive (probably "C:" on Disk 0). Right-click it →
   **Shrink Volume**.
3. In the dialog, set **Enter the amount of space to shrink in MB** to
   `250000` (= 250 GB). Adjust upward if you have more space and want a
   bigger Ubuntu install.
4. Click **Shrink**. This is non-destructive — your Windows files stay put.
5. After it finishes you'll see ~250 GB of **Unallocated** space at the end
   of Disk 0. Leave it as unallocated — don't create a partition for it.
   Ubuntu's installer will format it.

Common gotcha: Windows can only shrink up to where its "unmovable files"
sit on the disk. If the shrink dialog won't let you shrink as much as you
want, defragment the drive first (`defrag C: /D` in admin PowerShell) and
retry.

---

## Phase 3 — Install Ubuntu 22.04 LTS

### 3.1 Download the Ubuntu ISO

From https://ubuntu.com/download/desktop, grab **Ubuntu 22.04.X LTS Desktop**
(currently 22.04.5 or later — the .X point release doesn't matter).
File is ~5 GB.

### 3.2 Write the ISO to USB

Use **Rufus** (https://rufus.ie/), it's the simplest tool:

1. Plug in your 8+ GB USB stick (will be erased).
2. Open Rufus.
3. Device: select your USB.
4. Boot selection: click **SELECT** and pick the Ubuntu ISO.
5. Image option: **Standard Windows installation** — wait, no, that's for
   Windows. Just leave Rufus's defaults.
6. Partition scheme: **GPT** (modern systems).
7. Target system: **UEFI (non CSM)**.
8. Click **START**.
9. If prompted for ISOHybrid mode, choose **Write in ISO Image mode**.
10. Wait ~5 minutes. Leave the USB plugged in when it's done.

### 3.3 Reboot into the Ubuntu installer

1. Shut down Windows fully (not Restart — actually power down).
2. Power on and immediately tap the BIOS hotkey to interrupt boot. Common
   keys: **F2**, **F12**, **Del**, **Esc**. Your motherboard manual or boot
   screen will say which.
3. In the BIOS boot menu, select the USB drive.
4. You'll see the GRUB menu — pick **Try or Install Ubuntu**.

If your screen goes black or you see a `nouveau` error, that's the open-source
NVIDIA driver having trouble booting. Reboot, return to the GRUB menu, hit
`e` to edit the boot entry, find the line starting with `linux`, add
`nomodeset` to the end, then F10 to boot. You'll install Ubuntu first, then
install the proprietary NVIDIA driver in Phase 4.

### 3.4 Run the Ubuntu installer

Once Ubuntu boots from the USB, double-click **Install Ubuntu**.

1. **Welcome:** language = English (or whatever you prefer).
2. **Keyboard layout:** detect automatically, accept the suggestion.
3. **Updates and other software:**
   - **Normal installation** (selected by default)
   - **Download updates while installing Ubuntu** ✓
   - **Install third-party software for graphics and Wi-Fi hardware...** ✓
     ← VERY IMPORTANT. This installs the NVIDIA proprietary driver during
     setup so you don't have to fight nouveau later.
4. **Installation type:** select **Install Ubuntu alongside Windows Boot
   Manager**. The installer detects the unallocated space you created in
   Phase 2 and uses it.
5. **Drag the partition divider** if asked — the default is usually fine.
6. **Time zone:** click your region on the map.
7. **Who are you:**
   - Your name: whatever
   - Computer name: something like `aaron-isaac` (no spaces)
   - Pick a username and a strong password
   - **Require my password to log in** ✓
8. Click **Install Now**. Goes 20–40 minutes.
9. When prompted: **Restart Now**. The installer will tell you to remove
   the USB drive; do so when prompted.

### 3.5 First boot

The system reboots into the **GRUB menu**, where you pick **Ubuntu** or
**Windows Boot Manager** each time. Ubuntu is the default; it auto-selects
after 10 seconds.

Pick **Ubuntu**, log in, and you should see the Ubuntu desktop. If you see
graphics glitches or the screen is fuzzy, that's the open-source nouveau
driver — Phase 4 fixes it.

---

## Phase 4 — Install NVIDIA proprietary driver + CUDA

### 4.1 Verify what driver is currently active

Open a terminal (Ctrl+Alt+T) and run:

```bash
nvidia-smi
```

If you ticked "Install third-party software" in Phase 3.4, this might already
work. You'd see your RTX 4060 Ti listed. Skip to Phase 4.3 if so.

If `nvidia-smi` says "command not found" or shows an error, the proprietary
driver isn't installed yet. Continue with 4.2.

### 4.2 Install the NVIDIA driver

```bash
sudo apt update && sudo apt upgrade -y
sudo ubuntu-drivers devices
```

This shows you available drivers with one marked `recommended`. Install it:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

You should see your RTX 4060 Ti, driver version, and CUDA version. If you do,
the hard part is done — you have full native GPU access.

### 4.3 Install CUDA toolkit

Required for compiling some Isaac Lab components and for PyTorch CUDA
support.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
```

Add CUDA to your PATH:

```bash
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version
```

Should print CUDA 12.6.

---

## Phase 5 — Install system dependencies, Miniconda, Isaac Sim, Isaac Lab

### 5.1 System libraries

```bash
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
    libxcb-xkb1 \
    libvulkan1 \
    vulkan-tools \
    libxt6 \
    libglu1-mesa
```

Verify Vulkan can see the GPU — this is the test that kept failing in WSL2:

```bash
vulkaninfo --summary
```

You should see your NVIDIA RTX 4060 Ti listed with
`deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU` near the top of the
Devices section. On native Linux this just works.

### 5.2 Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
exec bash
```

Accept the Anaconda channels' ToS (one-time):

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 5.3 Create the Isaac Lab conda env

```bash
conda create -n isaaclab python=3.10 -y
conda activate isaaclab
pip install --upgrade pip
```

### 5.4 Install Isaac Sim

Critical: use the `[all,extscache]` extras. The bare `isaacsim` metapackage
is a stub.

```bash
pip install "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com
```

Downloads ~10 GB. Goes for several minutes.

Smoke-test Isaac Sim before continuing:

```bash
python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('Isaac Sim OK'); app.close()"
```

First run takes 2–5 minutes (shader cache). You'll see Isaac startup
warnings scrolling — those are normal. Watch for `Isaac Sim OK` at the
end. **This is the same command that segfaulted under WSL2. On native
Linux it will succeed.** If it doesn't, stop and debug before moving on.

### 5.5 Install Isaac Lab

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

Takes 5–15 minutes. Installs rsl_rl, skrl, and the Isaac Lab Python package
into your conda env.

### 5.6 End-to-end smoke test

Run a tiny PPO training run on the bundled Cartpole task:

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 \
    --num_envs 64 \
    --headless \
    --max_iterations 20
```

You should see iteration logs scrolling, GPU usage on `nvidia-smi` (open a
second terminal and watch), and rewards climbing. If you see this, your
install is fully functional. **You've made it past the hard part.**

---

## Phase 6 — Set up leorover_isaac

### 6.1 Clone your repo

```bash
cd ~
git clone https://github.com/asharpie/leorover_isaac.git
cd leorover_isaac
```

### 6.2 Install as an editable package

```bash
conda activate isaaclab
pip install -e .
```

Verify:

```bash
python -c "import leorover_isaac; print(leorover_isaac.__file__)"
```

Should print `/home/<you>/leorover_isaac/leorover_isaac/__init__.py`.

### 6.3 Shell aliases for convenience

Add to `~/.bashrc`:

```bash
alias isaac='conda activate isaaclab && cd ~/IsaacLab'
alias lab='~/IsaacLab/isaaclab.sh -p'
alias leorov='cd ~/leorover_isaac'
alias train_headless='~/IsaacLab/isaaclab.sh -p ~/IsaacLab/source/standalone/workflows/rsl_rl/train.py --headless'
```

`source ~/.bashrc` to apply.

You're now ready to start on Phase 1 of [PORTING_ROADMAP.md](PORTING_ROADMAP.md)
— the asset port. Welcome to native Linux RL.

---

## Accessing your PyBullet repo from Ubuntu

Two options. Pick whichever feels right.

**Option A — Mount the Windows partition.** Ubuntu can read your Windows
NTFS partition without any setup; it auto-mounts under `/media/<user>/`
when you open the Files app and click on it. Your PyBullet files at
`C:\Users\Aaron\Downloads\leoroverpybullet_share - Checkpoint Working...`
will be readable from Ubuntu at something like
`/media/aaron/Windows/Users/Aaron/Downloads/leoroverpybullet_share - Checkpoint Working...`.
Read-only by default unless you explicitly mount as read-write.

**Option B — Just git clone.** If your PyBullet repo is on GitHub, clone
it natively into Ubuntu (`cd ~ && git clone <your-repo-url>`). Cleaner,
fully writable, no Windows entanglement.

I'd suggest B for active work, A for "I just need to grab one file."

---

## Troubleshooting

### Screen goes black on first boot after install
The proprietary NVIDIA driver isn't loaded yet. Reboot, hold Shift to
force the GRUB menu, pick **Advanced options for Ubuntu**, then **(recovery
mode)** for the latest kernel. From the recovery menu pick **resume**, log
in at the text console, and run Phase 4.2 to install the driver. Reboot
into the normal desktop.

### `nvidia-smi` works but training crashes with "no CUDA-capable device"
Your conda env's PyTorch was installed without CUDA support. Reinstall:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

Check `python -c "import torch; print(torch.cuda.is_available())"` returns `True`.

### Grub menu doesn't appear at boot, Windows boots immediately
Boot from the Ubuntu install USB again, select **Try Ubuntu**, then run
`boot-repair` (you may need to install it via `sudo apt install boot-repair`
once Ubuntu is running). Use the **Recommended repair** option.

### Ubuntu won't connect to Wi-Fi
The driver wasn't installed. Use ethernet for now, then `sudo apt install
firmware-iwlwifi` (Intel cards) or `sudo apt install bcmwl-kernel-source`
(Broadcom) — check your wireless chipset with `lspci | grep -i wireless`.

### How do I switch back to Windows
Reboot. At the GRUB menu pick **Windows Boot Manager**. Or set Windows
as the default boot target by editing GRUB — see https://askubuntu.com/q/91754.

### My partition setup got weird and I want to undo everything
Boot into Windows, run **Disk Management**, identify the Ubuntu partition
(it'll show as "Healthy (Primary Partition)" with no drive letter, ~250 GB
size), right-click → **Delete Volume**. Then right-click your C: drive
→ **Extend Volume** to claim the freed space back.

You'll also want to remove the GRUB bootloader from Windows: open admin
PowerShell and run `bcdedit /set "{bootmgr}" path \EFI\Microsoft\Boot\bootmgfw.efi`.

---

## Where to go next

1. [PORTING_ROADMAP.md](PORTING_ROADMAP.md) — the actual port plan, phase by phase
2. [leorover_isaac/envs/README.md](leorover_isaac/envs/README.md) — PyBullet → Isaac Lab env mapping
3. [SETUP_HISTORY.md](SETUP_HISTORY.md) — the WSL2/Docker debugging story, for reference

Welcome to Isaac Lab.
