#!/bin/bash -l
# The 40%-sky survey (168 tiles, forecast noise): the catalogue models (published style, calibrated at one point,
# calibrated through the noise level; the calibrations measured on the 41-tile survey), then the map-level
# simulations (100 mocks, 500 prior draws, 10^4 training). Everything resumes, so a second submission continues
# where the first stopped. Submit (with the user's approval), e.g. twice, the second with --dependency=afterany:<first>:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 --mem=96G \
#        -t 04:00:00 -J mapsim-s40 -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/sims40.sh
set -o pipefail
TILES=${TILES:-168}
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export COSMOCNC_DATA=/mnt/home/bbolliet/ref/cosmocnc_jax/cosmocnc_jax/data
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
PY=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
cd $CODE
step() { echo; echo "=== $(date +%T) $*"; }
quiet() { grep --line-buffered -v -i "warn\|Constant folding\|dot_general\|necessarily a bug\|file a bug\|slow_operation_alarm\|operation took\|^  [0-9]*\. \|=====\|Top 10 largest\|Total ByteSize\|failed to serialize\|  -> \|spectra \[\|calibration (z_upper"; }
check() { [ "$1" -eq 0 ] || { echo "step failed (exit $1); stopping"; exit "$1"; }; }
step "node"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; cat $CODE/SNAPSHOT
for v in forecast41 calibf41 calibcf41; do
  for s in truth heldout train; do
    step "catalogue model $v, $TILES tiles: $s"
    PYTHONPATH=$CODE $PY benchmarks/mapsim/catalogue_trainset.py --variant $v --tiles $TILES --set $s 2>&1 | quiet | grep -v "/10000 done"; check ${PIPESTATUS[0]}
  done
done
for s in truth heldout train; do
  step "map-level simulations, $TILES tiles, forecast noise: $s"
  PYTHONPATH=$CODE $PY benchmarks/mapsim/maps_trainset.py --set $s --noise forecast --tiles $TILES 2>&1 | quiet | awk '/\/10000 \(/ {n++; if (n % 10) next} {print; fflush()}'; check ${PIPESTATUS[0]}
done
step "done"
