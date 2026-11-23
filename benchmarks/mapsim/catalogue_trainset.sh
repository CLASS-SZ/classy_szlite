#!/bin/bash -l
# The catalogue-level NPE simulations (catalogue_trainset.py): 100 at the true point, 500 held-out prior
# draws and 10^4 training draws. Submit (with the user's approval) from a frozen snapshot, e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 --mem=64G \
#        -t 01:00:00 -J mapsim-trainset -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/catalogue_trainset.sh
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export COSMOCNC_DATA=/mnt/home/bbolliet/ref/cosmocnc_jax/cosmocnc_jax/data
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
PY=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
BATCH=${BATCH:-16}
cd $CODE
step() { echo; echo "=== $(date +%T) $*"; }
step "node"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; cat $CODE/SNAPSHOT
for s in truth heldout train; do
  step "set $s"
  PYTHONPATH=$CODE $PY benchmarks/mapsim/catalogue_trainset.py --set $s --batch $BATCH 2>&1 | grep --line-buffered -v -i warn
  rc=${PIPESTATUS[0]}; [ $rc -eq 0 ] || { echo "set $s failed (exit $rc); stopping"; exit $rc; }
done
step "done"
