#!/bin/bash -l
# The catalogue model calibrated through the finder's noise level (variant (iv)), forecast noise by default, from the
# 25 points already measured (qcal_grid.py run): tests; the fit (qcal_grid.py fit --model conf); the catalogue sets
# (catalogue_trainset.py --variant calibcf41); the posterior, applied to the 100 map-level mocks and 500 prior draws.
# Submit (with the user's approval):
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 --mem=96G \
#        -t 01:30:00 -J mapsim-calc -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/calib_conf.sh
set -o pipefail
NOISE=${NOISE:-forecast}
if [ "$NOISE" = forecast ]; then VAR=calibcf41; MAPS=maps_npe_forecast; else VAR=calibc41; MAPS=maps_npe; fi
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export COSMOCNC_DATA=/mnt/home/bbolliet/ref/cosmocnc_jax/cosmocnc_jax/data
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-16}
PY=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
PYSBI=/mnt/home/bbolliet/venvs/sbi/bin/python
cd $CODE
step() { echo; echo "=== $(date +%T) $*"; }
quiet() { tr '\r' '\n' | grep --line-buffered -v -i "warn\|Constant folding\|dot_general\|necessarily a bug\|file a bug\|slow_operation_alarm\|operation took\|^  [0-9]*\. \|=====\|Top 10 largest\|Total ByteSize\|failed to serialize\|  -> \|Epochs trained:\|Drawing .* samples\|it/s\]\|s/it\]\|^                    \|rejection sampling\|max_sampling_time\|switching to\|accepted. It may\|proposal samples\|sample_batched"; }
check() { [ "$1" -eq 0 ] || { echo "step failed (exit $1); stopping"; exit "$1"; }; }
step "node"; hostname; cat $CODE/SNAPSHOT
step "1. catalogue tests"
PYTHONPATH=$CODE $PY -m pytest -q -p no:cacheprovider -s tests/test_sim_catalogue.py 2>&1 | quiet | grep -v "^\s*$" | tail -18; check ${PIPESTATUS[0]}
step "2. the calibration through the noise level, from the 25 points"
PYTHONPATH=$CODE $PY benchmarks/mapsim/qcal_grid.py fit --model conf --noise $NOISE 2>&1 | quiet; check ${PIPESTATUS[0]}
for s in truth heldout train; do
  step "3. catalogue model calibrated through the noise level ($VAR): $s"
  PYTHONPATH=$CODE $PY benchmarks/mapsim/catalogue_trainset.py --variant $VAR --set $s 2>&1 | quiet | grep -v "/10000 done\|spectra \["; check ${PIPESTATUS[0]}
done
step "4. its posterior, applied to the map-level mocks and prior draws"
$PYSBI benchmarks/mapsim/catalogue_npe.py --data $MAPSIM_OUT/catalogue_npe/$VAR --prior maps --tag t2calc --samples 4000 \
    --apply maps=$MAPSIM_OUT/$MAPS/truth.npz --apply mapheld=$MAPSIM_OUT/$MAPS/heldout.npz 2>&1 | quiet; check ${PIPESTATUS[0]}
step "done"
