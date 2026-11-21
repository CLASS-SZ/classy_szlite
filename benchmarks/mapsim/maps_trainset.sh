#!/bin/bash -l
# Map-level simulations of the 10%-sky survey (maps_trainset.py): the lensing tests (the stacked p
# now covers every confirmed q > 5 detection), then 100 mocks at the fiducial point, 500 held-out
# prior draws and 10^4 training draws. Resumable. Submit (with the user's approval), e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 --mem=96G \
#        -t 03:00:00 -J mapsim-maps -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/maps_trainset.sh
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
PY=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
cd $CODE
step() { echo; echo "=== $(date +%T) $*"; }
quiet() { grep --line-buffered -v -i "warn\|Constant folding\|dot_general\|necessarily a bug\|file a bug\|slow_operation_alarm\|operation took"; }
step "node"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; cat $CODE/SNAPSHOT
step "lensing tests"
PYTHONPATH=$CODE $PY -m pytest -q -p no:cacheprovider tests/test_sim_lensing.py 2>&1 | quiet | tail -4
rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] || { echo "tests failed (exit $rc); stopping"; exit $rc; }
for s in truth heldout train; do
  step "set $s"
  PYTHONPATH=$CODE $PY benchmarks/mapsim/maps_trainset.py --set $s 2>&1 | quiet
  rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] || { echo "set $s failed (exit $rc); stopping"; exit $rc; }
done
step "done"
