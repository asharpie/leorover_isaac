#!/usr/bin/env bash
# Convenience launcher for the UA lab box (Isaac Sim 4.5 standalone on driver 595).
# Bundles the three environment fixes the 4.5 stack needs here, then calls
# isaaclab.sh -p with whatever you pass. Examples:
#
#   scripts/run_lab.sh scripts/diag_drive.py --task Isaac-LeoRover-Flat-v0 --num_envs 16 --headless
#   scripts/run_lab.sh scripts/train.py --task Isaac-LeoRover-Mars-v0 --num_envs 4096 --headless
#
# Overridable via env: LEOROVER_ISAACLAB (path to the 4.5 IsaacLab), DISPLAY, XAUTHORITY.
set -euo pipefail

# 1. Expose the driver's libcuda.so under the unversioned name PhysX GPU dlopen()s.
#    User-local symlink only — never touches the system driver.
CUDALINK="$HOME/.local/cudalink"
if [ ! -e "$CUDALINK/libcuda.so" ]; then
  REAL=$(ldconfig -p | grep -m1 'libcuda.so.1' | awk '{print $NF}' || true)
  mkdir -p "$CUDALINK"
  if [ -n "${REAL:-}" ]; then ln -sf "$REAL" "$CUDALINK/libcuda.so"; fi
fi
export LD_LIBRARY_PATH="$CUDALINK:${LD_LIBRARY_PATH:-}"

# 2. Give the RTX/Vulkan renderer a display, or it errors "No device could be created"
#    even in --headless (it still enumerates the GPU through X).
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

# 3. The lab's Isaac Sim 4.5 IsaacLab (its _isaac_sim symlink points at the 4.5 standalone).
LAB="${LEOROVER_ISAACLAB:-/home/irl/Desktop/Core_libraries/NVIDIA_GPU/IsaacLab}"

echo "[run_lab] DISPLAY=$DISPLAY  IsaacLab=$LAB"
exec "$LAB/isaaclab.sh" -p "$@"
