# Setup history — why dual-boot, not WSL2

This document records what we tried before settling on dual-boot Ubuntu,
the symptoms we saw, and why each path failed. It's here for two reasons:

1. **You might want to revisit WSL2 someday** — once Sharpie's Windows
   build leaves the Insider channel and settles on stable 24H2+ with a
   matching NVIDIA driver, WSL2 should work and would be a more convenient
   workflow than dual-boot. This doc captures what to verify before retrying.
2. **An advisor or collaborator might ask why this isn't using WSL2 like
   "everyone else."** The answer is "we tried and validated that it doesn't
   work on this specific machine."

---

## Target system

- **OS:** Windows 11, build 10.0.26200.8457 (Canary/Dev Channel Insider)
- **GPU:** NVIDIA GeForce RTX 4060 Ti (8 GB)
- **Drivers tested:** NVIDIA Studio 610.43.02, then Game Ready 610.47
- **WSL:** WSL 2.7.3.0, kernel 6.6.114.1-1, WSLg 1.0.73
- **Ubuntu in WSL:** 22.04 LTS, fresh install

---

## Path 1 — Bare WSL2

Followed the standard "WSL2 + Isaac Lab" path:

1. `wsl --install -d Ubuntu-22.04` → succeeded
2. `nvidia-smi` inside Ubuntu → succeeded, GPU detected
3. Conda env + Isaac Sim pip install with `[all,extscache]` → succeeded
4. Smoke test (`SimulationApp({'headless': True})`) → **failed with
   Vulkan error**:

```
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed.
Vulkan 1.1 is not supported, or your driver requires an update.
Fatal Python error: Segmentation fault
```

### Diagnosis

`vulkaninfo --summary` showed only `llvmpipe` (CPU Vulkan), not the NVIDIA
GPU. Looking at `/usr/lib/wsl/lib/`:

```
libnvidia-encode.so       ← video encoder
libnvidia-gpucomp.so      ← CUDA compute
libnvidia-ml.so.1         ← NVML
libnvidia-ngx.so.1        ← DLSS
libnvidia-opticalflow.so  ← optical flow
nvidia-smi
```

**The OpenGL/Vulkan libraries weren't exposed.** Missing entries that should
be there for graphics passthrough:

- `libGLX_nvidia.so.0` (OpenGL entry point)
- `libnvidia-glcore.so.<ver>` (OpenGL core)
- `libnvidia-vulkan.so` or equivalent
- `libEGL_nvidia.so.0`

No amount of writing a `nvidia_icd.json` file fixed this, because there's
no NVIDIA Vulkan library for the JSON to point at.

### Things we tried that didn't fix it

- `wsl --update && wsl --shutdown` — no change to library list
- Reinstalled NVIDIA driver (Studio → Game Ready Driver 610.47, clean
  install option) — no change to library list
- Full Windows reboot after each driver install — no change
- Enabled Hardware-accelerated GPU Scheduling in Windows Settings — the
  registry key `HKLM:\...\GraphicsDrivers\HwSchMode` still didn't appear,
  even after reboot, suggesting the Insider build's Settings UI may not
  be persisting the toggle correctly

### Verdict

WSL2 graphics passthrough on Windows Insider 26200.x with NVIDIA driver
610.x is broken. The compute path works (CUDA libraries are passed through),
the graphics path doesn't. This appears to be a known pattern on Insider
builds — the WDDM interface evolves faster than NVIDIA's driver releases
catch up.

---

## Path 2 — Docker on WSL2 (via nvidia-container-toolkit)

Reasoning: NVIDIA officially recommends Docker for Isaac Sim, and the
`nvidia-container-toolkit` is supposed to handle GPU passthrough through
its own channel. Hope was that it would bypass WSL2's broken passthrough.

1. Installed Docker Desktop on Windows with WSL2 integration
2. Installed `nvidia-container-toolkit` in WSL
3. `docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi` →
   succeeded, GPU visible
4. Pulled `nvcr.io/nvidia/isaac-sim:4.5.0` (~16 GB)
5. Ran Isaac Sim smoke test inside container → **failed with identical
   Vulkan error and identical segfault location to the bare WSL2 attempt**

### Diagnosis

`nvidia-container-toolkit` on Docker Desktop for Windows doesn't have its
own GPU passthrough channel — it uses WSL2's GPU passthrough underneath.
So if WSL2 can't see the NVIDIA Vulkan libraries, neither can a container
running in WSL2, no matter what runtime mounts the toolkit adds.

### Verdict

Docker doesn't help when the underlying WSL2 graphics passthrough is broken.
This rules out the "use NVIDIA's official Isaac Sim Docker container" path
entirely on this machine.

---

## Path 3 — Dual-boot Ubuntu 22.04 (considered, deferred)

Reasoning: native Linux has direct hardware access. No virtualization layer
to translate between Windows and Linux GPU views. NVIDIA's drivers on
Ubuntu are well-tested and Isaac Lab is developed against this exact
configuration.

Trade-off: must reboot to switch between Windows and Linux. Mutually
exclusive — can't run Isaac training on Ubuntu and a Windows game (e.g.
Rocket League) at the same time. For Sharpie, who often plays Rocket
League between or during PyBullet training runs, this constraint mattered.

Decision: dual-boot guide is written and ready ([INSTALL.md](INSTALL.md))
in case the cloud path ever becomes impractical, but Path 4 (cloud) was
chosen instead because it preserves the gaming-during-training workflow.

---

## Path 4 — Cloud GPU rental on RunPod (chosen)

Reasoning: dual-boot's "one OS at a time" constraint conflicted with the
desire to keep gaming + training going simultaneously. Cloud GPU sidesteps
this entirely — training runs remotely on rented hardware in a data center,
while the local Windows machine keeps doing whatever else.

Trade-offs:

- **Ongoing cost** (~$10–40/month part-time) instead of $0 marginal cost
  of local hardware
- **Internet dependency** — no offline work, can't train on the train
- **~50–100 ms SSH latency** — invisible for training, noticeable only
  for any interactive GUI work in Isaac Sim (which we won't be doing much
  of anyway)
- **First-time SSH learning curve** — small, novel if you've never used SSH

In return:

- Local Windows install stays exactly as it is, including all games
- No disk partitioning risk
- Training runs in parallel with whatever else the laptop is doing
- Volume disk preserves work-in-progress across pod restarts
- RunPod community templates skip the Isaac Sim install entirely

This is the validated chosen path. See [CLOUD_SETUP.md](CLOUD_SETUP.md)
for the step-by-step.

---

---

## When to retry WSL2

If Sharpie's Windows install ever lands on a **stable** build (24H2 or
later non-Insider) with a **stable** NVIDIA driver release (something that
appears in NVIDIA's regular Studio/Game Ready cadence rather than a beta
or a number ahead of what's on nvidia.com's main page), it's worth trying
WSL2 again. The signal that WSL2 is healthy on a given setup is:

```bash
ls /usr/lib/wsl/lib/ | grep -i nvidia
```

If that includes `libGLX_nvidia.so.0` (or similar OpenGL/Vulkan libraries)
alongside the CUDA compute libraries, the passthrough is working and WSL2
+ Isaac Lab will install cleanly via the old INSTALL.md flow (preserved in
git history before the dual-boot rewrite). If only the compute libraries
show up, don't bother — same wall as before.

---

## Other paths considered but not tried

- **Cloud GPU rental** (Lambda Labs, Vast.ai, RunPod): would have worked
  immediately, but Sharpie wanted a local setup for iteration speed and to
  avoid ongoing rental cost. Stays on the table if dual-boot ever becomes
  impractical.
- **Older Windows build rollback**: could in principle revert Windows 11 to
  pre-Insider, but rolling back Windows is risky and slow. Dual-boot is the
  more durable answer.
- **Switching to AMD GPU**: would dodge the NVIDIA-specific passthrough
  issues but Isaac Lab is NVIDIA-only (PhysX requires CUDA), so this isn't
  a real option.
