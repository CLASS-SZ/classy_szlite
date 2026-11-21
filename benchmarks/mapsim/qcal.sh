#!/bin/bash -l
# The finder's q against the catalogue model's q on the map-level mocks (qcal.py). Submit (with the
# user's approval), e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 --mem=96G \
#        -t 00:45:00 -J mapsim-qcal -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/qcal.sh
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
PY=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
cd $CODE
quiet() { grep --line-buffered -v -i "warn\|Constant folding\|dot_general\|necessarily a bug\|file a bug\|slow_operation_alarm\|operation took\|^  [0-9]*\. \|=====\|Top 10 largest\|Total ByteSize\|failed to serialize\|  -> "; }
echo "=== $(date +%T) node"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; cat $CODE/SNAPSHOT
echo "=== $(date +%T) per-halo table (${NOISE:-mss2} noise)"
PYTHONPATH=$CODE $PY benchmarks/mapsim/qcal.py run --noise ${NOISE:-mss2} --mocks ${MOCKS:-20} 2>&1 | quiet; rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] || { echo "run failed ($rc)"; exit $rc; }
echo "=== $(date +%T) analysis"
PYTHONPATH=$CODE $PY benchmarks/mapsim/qcal.py analyse --noise ${NOISE:-mss2} 2>&1 | quiet; rc=${PIPESTATUS[0]}
echo "=== $(date +%T) done ($rc)"; exit $rc
