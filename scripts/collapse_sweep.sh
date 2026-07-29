#!/usr/bin/env bash
# collapse_sweep.sh - map the collapse curve: paired hybrid/LQR eval at rising harshness H.
#
# H scales geometry + soil severity together; friction protocol and soil GEOGRAPHY stay
# identical to the headline suites so the curve has one interpretable axis:
#   TERRAIN_AMP = 1.5*H          (hill relief)
#   SOIL CRR/CSINK/CSLIP/CLAT/KDIG *= H   (drag, dig-in, thrust decay, shear)
#   SOIL KREC /= H               (slower recovery when bogged)
#   SPEAK, SAND, TERRAIN_SEED, mu-range: UNCHANGED (same slip peak, same zone map, same world)
#
# Usage:  nohup bash scripts/collapse_sweep.sh > ~/leo_logs/collapse_$(date +%m%d).log 2>&1 &
#         bash scripts/collapse_sweep.sh 1.0 1.5 2.0          # custom H list
# Then:   python3 scripts/collapse_curve.py evals/collapse_manifest_<stamp>.txt
#
# ~30 min per point (n=4000 paired scenarios x 2 legs). Do NOT run while another
# GPU job (training / 90k suite) is active.
set -u
cd "$(dirname "$0")/.."

HLIST=("$@")
[ ${#HLIST[@]} -eq 0 ] && HLIST=(1.00 1.25 1.50 2.00 2.50 3.00)
NSCEN="${COLLAPSE_N:-4000}"

stamp="$(date +%Y%m%d_%H%M)"
manifest="evals/collapse_manifest_${stamp}.txt"
mkdir -p evals
echo "# H  evals_dir  AMP  (n=${NSCEN}/point, seed=world default, mu protocol default)" > "$manifest"
echo "[collapse] H points: ${HLIST[*]}  n/point=${NSCEN}  manifest=${manifest}"

for H in "${HLIST[@]}"; do
  AMP=$(awk -v h="$H" 'BEGIN{printf "%.3f", 1.5*h}')
  export LEOROVER_TERRAIN_AMP="$AMP"
  export LEOROVER_SOIL_CRR=$(awk  -v h="$H" 'BEGIN{printf "%.4f", 0.04*h}')
  export LEOROVER_SOIL_CSINK=$(awk -v h="$H" 'BEGIN{printf "%.4f", 0.30*h}')
  export LEOROVER_SOIL_CSLIP=$(awk -v h="$H" 'BEGIN{printf "%.4f", 0.50*h}')
  export LEOROVER_SOIL_CLAT=$(awk -v h="$H" 'BEGIN{printf "%.4f", 0.35*h}')
  export LEOROVER_SOIL_KDIG=$(awk -v h="$H" 'BEGIN{printf "%.4f", 0.50*h}')
  export LEOROVER_SOIL_KREC=$(awk -v h="$H" 'BEGIN{printf "%.4f", 0.25/h}')

  before=$(ls -1dt evals/*/ 2>/dev/null | head -1 || true)
  echo ""
  echo "[collapse] ===== H=${H}  AMP=${AMP}  CSINK=${LEOROVER_SOIL_CSINK} CSLIP=${LEOROVER_SOIL_CSLIP} ====="
  bash scripts/leo.sh eval --n "$NSCEN"
  after=$(ls -1dt evals/*/ 2>/dev/null | head -1 || true)

  if [ -n "$after" ] && [ "$after" != "$before" ] && [ -f "${after}/hybrid.csv" ] && [ -f "${after}/lqr.csv" ]; then
    echo "$H  ${after%/}  $AMP" >> "$manifest"
    echo "[collapse] H=${H} -> ${after%/}"
  else
    echo "$H  FAILED  $AMP" >> "$manifest"
    echo "[collapse] H=${H} FAILED (no new evals dir with both CSVs) - continuing"
  fi
done

echo ""
echo "[collapse] sweep done. Curve:"
python3 scripts/collapse_curve.py "$manifest" || true
