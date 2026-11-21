#!/bin/bash -l
# The catalogue-level simulator (sim/catalogue.py): its tests, the code test against cosmocnc_jax,
# the expected data vector and timing. Submit (with the user's approval) from a frozen snapshot:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p genx -C rome -c 16 --mem=32G -t 01:00:00 \
#        -J mapsim-cat -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/catalogue_check.sh
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export COSMOCNC_DIR=/mnt/home/bbolliet/ref/cosmocnc_jax/cosmocnc_jax
export COSMOCNC_DATA=$COSMOCNC_DIR/data
export JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
PY=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
cd $CODE
step() { echo; echo "=== $(date +%T) $*"; }
step "node"; hostname; cat $CODE/SNAPSHOT
step "1. tests"
PYTHONPATH=$CODE $PY -m pytest -q -p no:cacheprovider -s tests/test_sim_catalogue.py 2>&1 | grep --line-buffered -v -i warn; rc1=${PIPESTATUS[0]}
step "2. against cosmocnc_jax, expected data vector, timing"
PYTHONPATH=$CODE $PY benchmarks/mapsim/catalogue_vs_cosmocnc.py 2>&1 | grep --line-buffered -v -i warn; rc2=${PIPESTATUS[0]}
step "done (exits $rc1, $rc2)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
