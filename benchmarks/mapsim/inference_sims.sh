#!/bin/bash -l
# Simulations for the 10%-sky bias test: the catalogue tests, the catalogue model of the 41-tile survey
# (catalogue_trainset.py --variant survey41: truth, held-out, training), then the 10^4 map-level training
# simulations (maps_trainset.py --set train; resumable). Submit (with the user's approval), e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 --mem=96G \
#        -t 02:30:00 -J mapsim-isims -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/inference_sims.sh
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export COSMOCNC_DATA=/mnt/home/bbolliet/ref/cosmocnc_jax/cosmocnc_jax/data
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
PY=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
cd $CODE
step() { echo; echo "=== $(date +%T) $*"; }
quiet() { grep --line-buffered -v -i "warn\|Constant folding\|dot_general\|necessarily a bug\|file a bug\|slow_operation_alarm\|operation took\|^  [0-9]*\. \|=====\|Top 10 largest\|Total ByteSize\|failed to serialize\|  -> "; }
check() { [ "$1" -eq 0 ] || { echo "step failed (exit $1); stopping"; exit "$1"; }; }
step "node"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; cat $CODE/SNAPSHOT
step "1. catalogue tests"
PYTHONPATH=$CODE $PY -m pytest -q -p no:cacheprovider -s tests/test_sim_catalogue.py 2>&1 | quiet | grep -v "^\s*$" | tail -12; check ${PIPESTATUS[0]}
for s in truth heldout train; do
  step "2. catalogue model of the 41-tile survey: $s"
  PYTHONPATH=$CODE $PY benchmarks/mapsim/catalogue_trainset.py --variant survey41 --set $s 2>&1 | quiet; check ${PIPESTATUS[0]}
done
step "3. map-level training simulations"
PYTHONPATH=$CODE $PY benchmarks/mapsim/maps_trainset.py --set train 2>&1 | quiet; check ${PIPESTATUS[0]}
step "done"
