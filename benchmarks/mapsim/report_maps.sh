#!/bin/bash -l
# Maps of one simulated tile for the report (report_maps.py). Submit (with the user's approval):
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=8 --mem=48G \
#        -t 00:20:00 -J mapsim-rmaps -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/report_maps.sh
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
cd $CODE; cat SNAPSHOT
PYTHONPATH=$CODE /mnt/home/bbolliet/venvs/classy_szlite/bin/python benchmarks/mapsim/report_maps.py 2>&1 \
  | grep --line-buffered -v -i "warn\|Constant folding\|^  [0-9]*\. \|=====\|Top 10 largest\|Total ByteSize\|failed to serialize\|  -> "
exit ${PIPESTATUS[0]}
