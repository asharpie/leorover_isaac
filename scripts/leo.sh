#!/usr/bin/env bash
# =============================================================================
# leo.sh - one-stop control for the Leo Rover Isaac Lab project.
#
# Wraps the long run_lab.sh / train.py / trace_episode.py commands (and the
# reward env-vars) behind simple subcommands, so anyone can train, watch, stop
# and evaluate runs without memorising flags. Run `leo help` for the menu.
#
# Setup (once):
#   chmod +x scripts/leo.sh
#   echo "alias leo='$HOME/leorover_work/leorover_isaac/scripts/leo.sh'" >> ~/.bashrc
#   source ~/.bashrc
# Then from anywhere: `leo train hybrid`, `leo watch`, etc.
# (No alias? just run `scripts/leo.sh <cmd>` from the repo.)
# =============================================================================
set -uo pipefail

# --- locate the repo (this script lives in <repo>/scripts/) ------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO" || { echo "cannot cd to repo $REPO" >&2; exit 1; }

LAUNCH="$REPO/scripts/run_lab.sh"          # the Isaac launcher (sets DISPLAY, libcuda, etc.)
LOGDIR="$HOME/leo_logs"; mkdir -p "$LOGDIR"
ISAAC_PY="$HOME/Desktop/Core_libraries/NVIDIA_GPU/IsaacLab/_isaac_sim/python.sh"
GPU_TOTAL_MIB=24564                        # RTX 4090

# --- the validated reward fix (2026-06-20: 0% -> ~50% success). --raw skips it.
FIX_ENV=(LEOROVER_W_EFFORT=0.05 LEOROVER_W_PROGRESS=150 LEOROVER_W_SMOOTH=0.1 LEOROVER_ENT_COEF=0.005)

# --- task alias -> (gym id, log/experiment folder) ---------------------------
task_id()  { case "$1" in hybrid) echo Isaac-LeoRover-Mars-Hybrid-v0;; ppo) echo Isaac-LeoRover-Mars-v0;; flat) echo Isaac-LeoRover-Flat-v0;; *) echo "";; esac; }
task_exp() { case "$1" in hybrid) echo leo_rover_mars_hybrid;; ppo) echo leo_rover_mars;; flat) echo leo_rover_flat;; *) echo "";; esac; }

latest_run()  { ls -1dt "logs/$1"/*/ 2>/dev/null | head -1; }       # newest run dir for an experiment
latest_ckpt() { ls -1t "${1%/}"/model_*.pt 2>/dev/null | head -1; } # newest checkpoint inside a run dir
latest_log()  { ls -1t "$LOGDIR"/*.log 2>/dev/null | head -1; }

g=$'\e[32m'; y=$'\e[33m'; r=$'\e[31m'; b=$'\e[1m'; x=$'\e[0m'
say()  { echo "${g}[leo]${x} $*"; }
warn() { echo "${y}[leo] $*${x}"; }
err()  { echo "${r}[leo] $*${x}" >&2; }

usage() {
cat <<EOF
${b}leo${x} - Leo Rover Isaac control

${b}TRAIN${x}
  leo train hybrid [--envs N] [--iters N] [--raw] [--fg]
  leo train ppo    [...]              pure-PPO (no LQR baseline)
      defaults: --envs 4096, the validated reward fix ON, runs in background + logged
      --raw  use stock config weights (no fix)     --fg  run in foreground (don't detach)

${b}MONITOR${x}
  leo watch          live-tail the most recent training log (Ctrl-C stops watching, not training)
  leo curve          show success% + reward + std over recent iterations
  leo gpu            GPU usage + which training processes are yours
  leo checkpoints    list saved checkpoints of the latest run

${b}EVALUATE${x}
  leo trace [N]      population eval of the latest hybrid checkpoint (N = model number, else newest)
                     reports how many of the rovers reach the goal + saves a top-down plot
  leo tb             launch TensorBoard for the latest run (prints the SSH tunnel command)

${b}CONTROL${x}
  leo stop           safely stop YOUR training (handles Isaac's SIGTERM-trap; never orphans the GPU)

${b}Typical session${x}
  leo train hybrid     # start the full fixed-reward run
  leo watch            # watch it boot + learn   (then Ctrl-C)
  leo curve            # is success% climbing and reward rising?
  leo trace            # evaluate the newest checkpoint at scale
  leo stop             # end it cleanly when done
EOF
}

# -----------------------------------------------------------------------------
cmd_train() {
  local alias="${1:-}"; [ $# -gt 0 ] && shift
  local gym exp; gym="$(task_id "$alias")"; exp="$(task_exp "$alias")"
  [ -z "$gym" ] && { err "unknown task '${alias:-}'  (use: hybrid | ppo | flat)"; exit 1; }
  local envs=4096 iters="" raw=0 fg=0
  while [ $# -gt 0 ]; do case "$1" in
    --envs)  envs="${2:?}"; shift 2;;
    --iters) iters="${2:?}"; shift 2;;
    --raw)   raw=1; shift;;
    --fg)    fg=1; shift;;
    *) err "unknown flag '$1'"; exit 1;;
  esac; done

  # GPU courtesy check (shared lab box - don't stomp a labmate's job)
  local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  say "GPU in use: ${used:-?} MiB / ${GPU_TOTAL_MIB} MiB"
  if [ -n "${used:-}" ] && [ "$used" -gt 3000 ]; then
    warn "the GPU already has >3 GB in use. If that's someone else's job, wait or ask - don't kill it."
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | sed 's/^/        /'
  fi

  local args=(--task "$gym" --num_envs "$envs" --headless)
  [ -n "$iters" ] && args+=(--max_iterations "$iters")
  local pre=(); [ "$raw" -eq 0 ] && pre=("${FIX_ENV[@]}")

  local log="$LOGDIR/${exp}_$(date +%Y%m%d_%H%M%S).log"
  say "task   : $gym"
  say "envs   : $envs    reward: $([ "$raw" -eq 1 ] && echo 'STOCK (--raw)' || echo 'FIXED')"
  say "log    : $log"
  warn "starting in 5s - Ctrl-C now to abort"; sleep 5

  if [ "$fg" -eq 1 ]; then
    env "${pre[@]}" "$LAUNCH" scripts/train.py "${args[@]}" 2>&1 | tee "$log"
  else
    nohup env "${pre[@]}" "$LAUNCH" scripts/train.py "${args[@]}" >"$log" 2>&1 &
    say "launched (PID $!). It keeps running if you log out."
    say "watch it:  leo watch        check progress:  leo curve        stop it:  leo stop"
  fi
}

cmd_watch() {
  local l; l="$(latest_log)"; [ -z "$l" ] && { err "no logs in $LOGDIR yet - start a run with 'leo train hybrid'"; exit 1; }
  say "tailing $l"; say "(Ctrl-C stops watching; training keeps going)"
  tail -n 40 -f "$l"
}

cmd_curve() {
  local l; l="$(latest_log)"; [ -z "$l" ] && { err "no logs yet"; exit 1; }
  say "run log: $l"
  echo "${b}-- ADR success% (latest 25) --${x}"
  grep -oE "success=[0-9.]+%" "$l" | tail -25 | nl | sed 's/^/   /' || true
  echo "${b}-- Mean reward (latest 15) --${x}"
  grep "Mean reward" "$l" | tail -15 | sed 's/^ */   /' || true
  echo "${b}-- Mean action noise std (latest 15)  [healthy ~0.2-0.7; ~0.02 = collapsed] --${x}"
  grep "Mean action noise std" "$l" | tail -15 | sed 's/^ */   /' || true
}

cmd_gpu() {
  nvidia-smi
  echo "${b}-- your training processes --${x}"
  ps -o pid,etime,cmd -u "$USER" 2>/dev/null | grep "scripts/train.py" | grep -v grep | sed 's/^/   /' || echo "   (none running)"
}

cmd_stop() {
  local pids; pids="$(pgrep -u "$USER" -f 'scripts/train.py' 2>/dev/null | tr '\n' ' ')"
  [ -z "${pids// /}" ] && { say "no training process of yours is running."; exit 0; }
  echo "${b}will force-stop these (yours only):${x}"
  ps -o pid,etime,cmd -p ${pids} 2>/dev/null | sed 's/^/   /'
  read -r -p "$(echo "${y}[leo] stop them? [y/N] ${x}")" ans
  case "${ans:-}" in
    y|Y|yes)
      pkill -9 -u "$USER" -f 'scripts/train.py'
      sleep 2
      say "stopped. GPU memory now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)";;
    *) say "left it running.";;
  esac
}

cmd_checkpoints() {
  local alias="${1:-hybrid}" run
  run="$(latest_run "$(task_exp "$alias")")"
  [ -z "$run" ] && { err "no runs yet for '$alias'"; exit 1; }
  say "latest $alias run: $run"
  if ls "${run%/}"/model_*.pt >/dev/null 2>&1; then
    ls -1t "${run%/}"/model_*.pt | sed 's/^/   /'
  else
    echo "   (no checkpoints yet - the first is saved at iteration 200)"
  fi
}

cmd_trace() {
  local model="" envs=1024 steps=1500
  while [ $# -gt 0 ]; do case "$1" in
    --envs)  envs="${2:?}"; shift 2;;
    --steps) steps="${2:?}"; shift 2;;
    [0-9]*)  model="$1"; shift;;
    *) err "unknown arg '$1'"; exit 1;;
  esac; done
  local run; run="$(latest_run leo_rover_mars_hybrid)"
  [ -z "$run" ] && { err "no hybrid runs yet"; exit 1; }
  local ckpt
  if [ -n "$model" ]; then ckpt="${run%/}/model_${model}.pt"; else ckpt="$(latest_ckpt "$run")"; fi
  [ -z "$ckpt" ] || [ ! -f "$ckpt" ] && { err "checkpoint not found: ${ckpt:-<none>} (try 'leo checkpoints')"; exit 1; }
  say "evaluating $ckpt"
  say "envs=$envs steps=$steps  (read the POPULATION block at the end, not env 0)"
  "$LAUNCH" scripts/trace_episode.py --task Isaac-LeoRover-Mars-Hybrid-v0 \
            --checkpoint "$ckpt" --num_envs "$envs" --steps "$steps"
  say "top-down plot: ${run%/}/eval_trace/trace.png"
  say "copy it to your laptop (run on the LAPTOP):"
  say "   scp irl@10.115.102.210:${run%/}/eval_trace/trace.png ."
}

cmd_tb() {
  local run; run="$(latest_run leo_rover_mars_hybrid)"; [ -z "$run" ] && { err "no runs yet"; exit 1; }
  [ ! -e "$ISAAC_PY" ] && { err "Isaac python not found at $ISAAC_PY"; exit 1; }
  say "TensorBoard for: $run"
  warn "on your LAPTOP, open another terminal and run the tunnel:"
  warn "   ssh -L 6006:localhost:6006 irl@10.115.102.210"
  warn "then browse http://localhost:6006   (Ctrl-C here stops TensorBoard)"
  "$ISAAC_PY" -m tensorboard.main --logdir "$run" --port 6006
}

# --- dispatch ----------------------------------------------------------------
case "${1:-help}" in
  train)              shift; cmd_train "$@";;
  watch)              cmd_watch;;
  curve|progress)     cmd_curve;;
  gpu|status)         cmd_gpu;;
  stop|kill)          cmd_stop;;
  checkpoints|ckpts)  shift; cmd_checkpoints "$@";;
  trace|eval)         shift; cmd_trace "$@";;
  tb|tensorboard)     cmd_tb;;
  help|-h|--help|"")   usage;;
  *) err "unknown command '$1'"; echo; usage; exit 1;;
esac
