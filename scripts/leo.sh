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

# --- the validated reward fix (2026-06-20: parked -> 59% success). --raw skips it.
FIX_ENV=(LEOROVER_W_EFFORT=${LEOROVER_W_EFFORT:-0.05} LEOROVER_W_PROGRESS=${LEOROVER_W_PROGRESS:-150} LEOROVER_W_SMOOTH=${LEOROVER_W_SMOOTH:-0.1})
# exploration + rollout, overridable per run with --ent / --rollout.
#   ent 0.001 (2026-06-21, down from 0.005): lets action-std fall so the policy sharpens
#     instead of failing ~40% of episodes from exploration noise -> higher success.
#   rollout 32 default; --rollout 64 gives the value fn more of the ~1400-step episode
#     so the terminal +200 success propagates (PyBullet used full-episode rollouts).
DEFAULT_ENT=0.001
BOX_HOST="irl@10.115.102.210"                 # how YOUR laptop reaches the box (for scp helpers)

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
      defaults: --envs 4096, reward fix ON, ent 0.001, rollout 32, background + logged
      --ent E       PPO entropy coef (lower = sharper / less noise; default 0.001)
      --rollout N   steps per update (higher = better long-horizon credit; default 32)
      --speed S     rover speed multiplier (1=crawl default, 4.8=full 0.4 m/s; fixes stagnation kills)
      --residual F  PPO residual authority (default 0.33 = stable; 1=full/unstable, 0=pure LQR)
      --raw  stock config (no fix)     --fg  foreground (don't detach)

${b}MONITOR${x}
  leo watch          live-tail the most recent training log (Ctrl-C stops watching, not training)
  leo curve          show success% + reward + std over recent iterations
  leo gpu            GPU usage + which training processes are yours
  leo checkpoints    list saved checkpoints of the latest run
  leo report         full performance report (success, progress, reward, CTE trends) — copy-pasteable
  leo csv            print the exact timestamped scp line to pull the latest CSV to your laptop

${b}EVALUATE${x}
  leo trace [N]      population eval of the latest hybrid checkpoint (N = model number, else newest)
                     reports how many of the rovers reach the goal + saves a top-down plot
  leo trace --lqr    same, but force the PPO residual to 0 = evaluate the bare LQR baseline
  leo diagnose [--terrain P] [--lqr]   classify WHY rovers stall (wedged/idle/off-path/slow) + plot
  leo eval [--paths discrete|random] [--friction F] [--n N] [--levels ..] [--envs N]
                     THE eval: all 3 controllers over the IDENTICAL episodes (same path+terrain+
                     pose+friction) -> paired t-test / McNemar / Cohen's d. Defaults: 9 discrete
                     geometries, 90k scenarios. --paths random = slope study. Usually just 'leo eval'.
  leo quickeval <hybrid|lqr|ppo> [--levels 0,20,..] [--envs N] [--steps N]
                     quick single-controller terrain sweep -> evals/<algo>_<ts>.csv (unpaired, fast)
  leo compare        side-by-side success/progress-by-terrain of the latest quickeval CSVs
  leo evalcsv <algo> print the scp line to pull a quickeval CSV to your laptop
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
# Deterministic held-out evaluation across a terrain sweep -> evals/<algo>_<ts>.csv
cmd_eval() {
  local alias="${1:-hybrid}"; [ $# -gt 0 ] && shift
  local levels="0,20,40,60,80,100" envs=1024 steps=6000   # row-exact for the 6-row bank (flat row + max row included)
  while [ $# -gt 0 ]; do case "$1" in
    --levels) levels="${2:?}"; shift 2;;
    --envs)   envs="${2:?}"; shift 2;;
    --steps)  steps="${2:?}"; shift 2;;
    *) err "unknown flag '$1'"; exit 1;;
  esac; done
  local task exp zero=""
  case "$alias" in
    hybrid) task="Isaac-LeoRover-Mars-Hybrid-v0"; exp="leo_rover_mars_hybrid";;
    lqr)    task="Isaac-LeoRover-Mars-Hybrid-v0"; exp="leo_rover_mars_hybrid"; zero="--zero-residual";;
    ppo)    task="Isaac-LeoRover-Mars-v0";        exp="leo_rover_mars";;
    *) err "unknown eval target '$alias'  (use: hybrid | lqr | ppo)"; exit 1;;
  esac
  local run; run="$(latest_run "$exp")"
  [ -z "$run" ] && { err "no '$exp' runs yet to evaluate (train one first)"; exit 1; }
  local ckpt; ckpt="$(latest_ckpt "$run")"
  { [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; } && { err "no checkpoint in ${run} (try 'leo checkpoints')"; exit 1; }
  local ts out; ts="$(date +%Y%m%d_%H%M%S)"; out="$REPO/evals/${alias}_${ts}.csv"; mkdir -p "$REPO/evals"
  local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  say "eval   : ${b}$alias${x}    task=$task"
  say "ckpt   : $ckpt"
  say "levels : $levels    envs=$envs steps=$steps"
  say "out    : $out"
  if [ -n "${used:-}" ] && [ "$used" -gt 3000 ]; then
    warn "GPU already has >3 GB in use - if a training job is mid-iteration, stop it first (eval needs the GPU)."
  fi
  warn "starting in 4s - Ctrl-C to abort"; sleep 4
  "$LAUNCH" scripts/evaluate_policy.py --task "$task" --checkpoint "$ckpt" \
            --levels "$levels" --num_envs "$envs" --steps "$steps" $zero --out "$out"
  echo
  say "done -> $out"
  say "pull to laptop:  ${b}leo evalcsv $alias${x}     compare all three:  ${b}leo compare${x}"
}

# Side-by-side comparison of the latest hybrid / lqr / ppo eval CSVs
cmd_compare() {
  local d="$REPO/evals" a f args=()
  [ ! -d "$d" ] && { err "no evals/ yet - run 'leo eval hybrid' (and lqr / ppo) first"; exit 1; }
  for a in hybrid lqr ppo; do
    f="$(ls -1t "$d/${a}_"*.csv 2>/dev/null | head -1)"
    [ -n "$f" ] && args+=("$a=$f")
  done
  [ ${#args[@]} -eq 0 ] && { err "no eval CSVs in $d - run 'leo eval <algo>' first"; exit 1; }
  say "comparing latest eval of: $(for _a in "${args[@]}"; do echo -n "${_a%%=*} "; done)"
  python3 "$REPO/scripts/eval_report.py" "${args[@]}"
}

# Print the scp line to pull an eval CSV to the laptop
cmd_evalcsv() {
  local alias="${1:-hybrid}" f
  f="$(ls -1t "$REPO/evals/${alias}_"*.csv 2>/dev/null | head -1)"
  [ -z "$f" ] && { err "no eval CSV for '$alias' yet (run 'leo eval $alias')"; exit 1; }
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  echo "${b}Copy-paste into PowerShell on your LAPTOP:${x}"
  echo
  echo "mkdir \$HOME\\Downloads\\leo_csvs -Force"
  echo "scp ${BOX_HOST}:$f \$HOME\\Downloads\\leo_csvs\\eval_${alias}_${ts}.csv"
  echo
  say "lands at  Downloads\\leo_csvs\\eval_${alias}_${ts}.csv  (latest $alias eval)"
}

# -----------------------------------------------------------------------------
cmd_train() {
  local alias="${1:-}"; [ $# -gt 0 ] && shift
  local gym exp; gym="$(task_id "$alias")"; exp="$(task_exp "$alias")"
  [ -z "$gym" ] && { err "unknown task '${alias:-}'  (use: hybrid | ppo | flat)"; exit 1; }
  # residual default respects a pre-set LEOROVER_RES_SCALE (it used to be silently
  # clobbered to 0.33 -- the 20260706_173712 run trained at 0.33 despite the env var).
  local envs=4096 iters="" raw=0 fg=0 ent="$DEFAULT_ENT" rollout="" speed=1 residual="${LEOROVER_RES_SCALE:-0.33}"
  while [ $# -gt 0 ]; do case "$1" in
    --envs)     envs="${2:?}"; shift 2;;
    --iters)    iters="${2:?}"; shift 2;;
    --ent)      ent="${2:?}"; shift 2;;
    --rollout)  rollout="${2:?}"; shift 2;;
    --speed)    speed="${2:?}"; shift 2;;
    --residual) residual="${2:?}"; shift 2;;
    --raw)      raw=1; shift;;
    --fg)       fg=1; shift;;
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
  local pre=()
  if [ "$raw" -eq 0 ]; then
    pre=("${FIX_ENV[@]}" "LEOROVER_ENT_COEF=$ent")
    [ -n "$rollout" ] && pre+=("LEOROVER_NUM_STEPS=$rollout")
  fi
  [ "$speed" != "1" ] && pre+=("LEOROVER_SPEED_SCALE=$speed")
  pre+=("LEOROVER_RES_SCALE=$residual")   # always pass it; default 0.33 (env default matches)

  local log="$LOGDIR/${exp}_$(date +%Y%m%d_%H%M%S).log"
  say "task   : $gym"
  say "envs   : $envs    reward: $([ "$raw" -eq 1 ] && echo 'STOCK (--raw)' || echo 'FIXED')"
  [ "$raw" -eq 0 ] && say "ent    : $ent    rollout(num_steps): ${rollout:-32}"
  say "speed  : ${speed}x  (1=crawl ~0.035 m/s, 4.8=full ~0.4 m/s; lifts rovers off the 0.02 stagnation kill-line)"
  say "residual: ${residual}x  (default 0.33 = capped PPO authority, stable + ~88-92%; 1=full, 0=pure LQR)"
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
  local model="" envs=1024 steps=1500 zr=""
  while [ $# -gt 0 ]; do case "$1" in
    --envs)  envs="${2:?}"; shift 2;;
    --steps) steps="${2:?}"; shift 2;;
    --zero-residual|--lqr) zr="--zero-residual"; shift;;
    [0-9]*)  model="$1"; shift;;
    *) err "unknown arg '$1'"; exit 1;;
  esac; done
  local run; run="$(latest_run leo_rover_mars_hybrid)"
  [ -z "$run" ] && { err "no hybrid runs yet"; exit 1; }
  local ckpt
  if [ -n "$model" ]; then ckpt="${run%/}/model_${model}.pt"; else ckpt="$(latest_ckpt "$run")"; fi
  [ -z "$ckpt" ] || [ ! -f "$ckpt" ] && { err "checkpoint not found: ${ckpt:-<none>} (try 'leo checkpoints')"; exit 1; }
  say "evaluating $ckpt"
  [ -n "$zr" ] && say "ZERO-RESIDUAL: pure LQR baseline (PPO output forced to 0)"
  say "envs=$envs steps=$steps  (read the POPULATION block at the end, not env 0)"
  "$LAUNCH" scripts/trace_episode.py --task Isaac-LeoRover-Mars-Hybrid-v0 \
            --checkpoint "$ckpt" --num_envs "$envs" --steps "$steps" $zr
  local png="$REPO/${run%/}/eval_trace/trace.png"   # absolute, so scp works from the laptop
  say "top-down plot: $png"
  say "copy it to your laptop (run this ON THE LAPTOP):"
  say "   scp irl@10.115.102.210:$png ."
}

# Diagnose WHY rovers stall (commanded-vs-actual velocity classification + plot)
cmd_diagnose() {
  local model="" terr=20 lqr=""
  while [ $# -gt 0 ]; do case "$1" in
    --terrain) terr="${2:?}"; shift 2;;
    --lqr|--zero-residual) lqr="--zero-residual"; shift;;
    [0-9]*)  model="$1"; shift;;
    *) err "unknown arg '$1'"; exit 1;;
  esac; done
  local run; run="$(latest_run leo_rover_mars_hybrid)"
  [ -z "$run" ] && { err "no hybrid runs yet"; exit 1; }
  local ckpt
  if [ -n "$model" ]; then ckpt="${run%/}/model_${model}.pt"; else ckpt="$(latest_ckpt "$run")"; fi
  { [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; } && { err "checkpoint not found (try 'leo checkpoints')"; exit 1; }
  say "diagnosing stalls in $ckpt"
  say "terrain=${terr}%   ${lqr:+(pure LQR baseline)}"
  "$LAUNCH" scripts/diagnose_stalls.py --task Isaac-LeoRover-Mars-Hybrid-v0 \
            --checkpoint "$ckpt" --terrain "$terr" $lqr
  local png="$REPO/${run%/}/stall_diag/stall_diag.png"
  echo
  say "read the VERDICT printed above; visual is at: $png"
  say "pull it (run ON THE LAPTOP):  scp ${BOX_HOST}:$png ."
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

cmd_report() {
  local alias="${1:-hybrid}" run
  run="$(latest_run "$(task_exp "$alias")")"
  [ -z "$run" ] && { err "no runs yet for '$alias'"; exit 1; }
  local csv="${run%/}/csv/episode_metrics.csv"
  [ ! -f "$csv" ] && { err "no episode_metrics.csv yet (first episodes still finishing)"; exit 1; }
  python3 "$REPO/scripts/leo_report.py" "$csv"
  echo
  say "for a full visual analysis, send Claude this file (run on your LAPTOP):"
  say "   scp irl@10.115.102.210:$REPO/${run%/}/csv/episode_metrics.csv ."
}

cmd_csv() {
  local alias="${1:-hybrid}" run
  run="$(latest_run "$(task_exp "$alias")")"
  [ -z "$run" ] && { err "no runs yet for '$alias'"; exit 1; }
  local csv="$REPO/${run%/}/csv/episode_metrics.csv"
  [ ! -f "$csv" ] && { err "no episode_metrics.csv yet"; exit 1; }
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  echo "${b}Copy-paste these two lines into PowerShell on your LAPTOP:${x}"
  echo
  echo "mkdir \$HOME\\Downloads\\leo_csvs -Force"
  echo "scp ${BOX_HOST}:$csv \$HOME\\Downloads\\leo_csvs\\episode_matrix_${ts}.csv"
  echo
  say "lands at  Downloads\\leo_csvs\\episode_matrix_${ts}.csv  (latest $alias run)"
}

# Paired, scenario-locked eval: all three controllers over the IDENTICAL episodes,
# then the paired statistics. This is the paper's matched-condition protocol.
cmd_pairedeval() {
  local paths="discrete" friction="1.0" nscen=90000 levels="0,20,40,60,80,100" envs=1024   # row-exact (old 10,30,50,70 collapsed onto rows 0/2/4)
  while [ $# -gt 0 ]; do case "$1" in
    --paths) paths="${2:?}"; shift 2;;
    --friction) friction="${2:?}"; shift 2;;
    --n|--scenarios) nscen="${2:?}"; shift 2;;
    --levels) levels="${2:?}"; shift 2;;
    --envs) envs="${2:?}"; shift 2;;
    *) err "unknown flag '$1'"; exit 1;;
  esac; done
  local hrun hckpt prun pckpt
  hrun="$(latest_run leo_rover_mars_hybrid)"; [ -z "$hrun" ] && { err "no hybrid run to eval (train one first)"; exit 1; }
  hckpt="$(latest_ckpt "$hrun")"; { [ -z "$hckpt" ] || [ ! -f "$hckpt" ]; } && { err "no hybrid checkpoint"; exit 1; }
  prun="$(latest_run leo_rover_mars)"; pckpt=""; [ -n "$prun" ] && pckpt="$(latest_ckpt "$prun")"
  local ts dir scen; ts="$(date +%Y%m%d_%H%M%S)"; dir="$REPO/evals/paired/$ts"; mkdir -p "$dir"; scen="$dir/scenarios.npz"
  say "paired eval: paths=${b}$paths${x} friction=$friction scenarios=$nscen levels=$levels envs=$envs"
  say "hybrid ckpt: $hckpt"
  [ -n "$pckpt" ] && say "ppo ckpt   : $pckpt" || warn "no pure-PPO run found -> comparing hybrid vs LQR only"
  local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  { [ -n "${used:-}" ] && [ "$used" -gt 3000 ]; } && warn "GPU already has >3 GB in use - stop other jobs first (eval needs the GPU)."
  warn "starting in 4s - Ctrl-C to abort"; sleep 4
  # 1) hybrid pass BUILDS + saves the shared scenario file
  "$LAUNCH" scripts/paired_eval.py --task Isaac-LeoRover-Mars-Hybrid-v0 --checkpoint "$hckpt" \
     --paths "$paths" --num_scenarios "$nscen" --levels "$levels" --friction "$friction" \
     --num_envs "$envs" --scenarios "$scen" --out "$dir/hybrid.csv" || { err "hybrid pass failed"; exit 1; }
  # 2) pure-LQR pass REUSES the scenario file (zero residual, same friction)
  "$LAUNCH" scripts/paired_eval.py --task Isaac-LeoRover-Mars-Hybrid-v0 --checkpoint "$hckpt" \
     --zero-residual --paths "$paths" --friction "$friction" --num_envs "$envs" \
     --scenarios "$scen" --out "$dir/lqr.csv" || { err "lqr pass failed"; exit 1; }
  local sargs=(hybrid="$dir/hybrid.csv" lqr="$dir/lqr.csv")
  # 3) pure-PPO pass (if a run exists), same scenarios + friction
  if [ -n "$pckpt" ]; then
    "$LAUNCH" scripts/paired_eval.py --task Isaac-LeoRover-Mars-v0 --checkpoint "$pckpt" \
       --paths "$paths" --friction "$friction" --num_envs "$envs" \
       --scenarios "$scen" --out "$dir/ppo.csv" && sargs+=(ppo="$dir/ppo.csv") || warn "ppo pass failed - reporting hybrid vs lqr only"
  fi
  echo; say "paired statistics:"
  python3 "$REPO/scripts/paired_stats.py" "${sargs[@]}" | tee "$dir/stats.txt"
  local made="hybrid.csv, lqr.csv, stats.txt"; [ -n "$pckpt" ] && made="hybrid.csv, lqr.csv, ppo.csv, stats.txt"
  echo; say "done -> $dir   ($made)"
  say "pull to laptop:  scp -r ${BOX_HOST}:$dir \$HOME\\Downloads\\leo_csvs\\"
}

# --- dispatch ----------------------------------------------------------------
case "${1:-help}" in
  train)              shift; cmd_train "$@";;
  watch)              cmd_watch;;
  curve|progress)     cmd_curve;;
  gpu|status)         cmd_gpu;;
  stop|kill)          cmd_stop;;
  checkpoints|ckpts)  shift; cmd_checkpoints "$@";;
  report|stats)       shift; cmd_report "$@";;
  csv|getcsv)         shift; cmd_csv "$@";;
  trace)              shift; cmd_trace "$@";;
  diagnose|stalls)    shift; cmd_diagnose "$@";;
  eval)               shift; cmd_pairedeval "$@";;
  quickeval|quick)    shift; cmd_eval "$@";;
  compare|cmp)        cmd_compare;;
  pairedeval|paired)  shift; cmd_pairedeval "$@";;
  evalcsv)            shift; cmd_evalcsv "$@";;
  tb|tensorboard)     cmd_tb;;
  help|-h|--help|"")   usage;;
  *) err "unknown command '$1'"; echo; usage; exit 1;;
esac
