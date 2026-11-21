#!/bin/bash -l
# The bias test with the calibrated catalogue model (variant (ii)): tests, the calibration from the per-halo
# table (qcal.py analyse -> qcal/calibration.npz), the calibrated catalogue sets of the 41-tile survey
# (catalogue_trainset.py --variant calib41), then the catalogue-level posterior trained on them, validated on
# its own held-out draws and applied to the map-level mocks (catalogue_npe.py, CPU torch).
# Submit (with the user's approval), e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 --mem=96G \
#        -t 02:00:00 -J mapsim-calib -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/calib_bias.sh
# NOISE=forecast: the forecast-noise survey (qcal_forecast, catalogue variant calibf41, maps_npe_forecast).
set -o pipefail
NOISE=${NOISE:-mss2}
if [ "$NOISE" = forecast ]; then VAR=calibf41; MAPS=maps_npe_forecast; else VAR=calib41; MAPS=maps_npe; fi
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
quiet() { tr '\r' '\n' | grep --line-buffered -v -i "warn\|Constant folding\|dot_general\|necessarily a bug\|file a bug\|slow_operation_alarm\|operation took\|^  [0-9]*\. \|=====\|Top 10 largest\|Total ByteSize\|failed to serialize\|  -> \|Epochs trained:\|Drawing .* samples\|it/s\]\|s/it\]"; }
check() { [ "$1" -eq 0 ] || { echo "step failed (exit $1); stopping"; exit "$1"; }; }
step "node"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; cat $CODE/SNAPSHOT
step "1. catalogue tests"
PYTHONPATH=$CODE $PY -m pytest -q -p no:cacheprovider -s tests/test_sim_catalogue.py 2>&1 | quiet | grep -v "^\s*$" | tail -14; check ${PIPESTATUS[0]}
if [ ! -f $MAPSIM_OUT/qcal$([ "$NOISE" = forecast ] && echo _forecast)/halos.npy ]; then
  step "2a. per-halo table on the map-level mocks ($NOISE noise)"
  PYTHONPATH=$CODE $PY benchmarks/mapsim/qcal.py run --noise $NOISE 2>&1 | quiet | tail -3; check ${PIPESTATUS[0]}
fi
step "2. calibration from the per-halo table"
PYTHONPATH=$CODE $PY benchmarks/mapsim/qcal.py analyse --noise $NOISE 2>&1 | quiet | sed -n '/^7\./,$p'; check ${PIPESTATUS[0]}
for s in truth heldout train; do
  step "3. calibrated catalogue model of the 41-tile survey: $s"
  PYTHONPATH=$CODE $PY benchmarks/mapsim/catalogue_trainset.py --variant $VAR --set $s 2>&1 | quiet | grep -v "/10000 done"; check ${PIPESTATUS[0]}
done
step "4. calibrated catalogue-level posterior, applied to the map-level mocks"
$PYSBI benchmarks/mapsim/catalogue_npe.py --data $MAPSIM_OUT/catalogue_npe/$VAR --prior maps --tag t2cal --samples 4000 \
    --apply maps=$MAPSIM_OUT/$MAPS/truth.npz --apply mapheld=$MAPSIM_OUT/$MAPS/heldout.npz 2>&1 | quiet; check ${PIPESTATUS[0]}
step "done"
