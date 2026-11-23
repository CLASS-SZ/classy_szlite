#!/bin/bash -l
# The catalogue model with a calibration that follows the parameters (variant (iii)), forecast noise by default:
# tests; a check that the reworked per-halo table still reproduces the mocks; the calibration at 25 points of the
# prior (qcal_grid.py run) and its linear fit (fit); the catalogue sets (catalogue_trainset.py --variant calibtf41);
# the posterior trained on them, applied to the 100 map-level mocks and the 500 map-level prior draws.
# Submit (with the user's approval):
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 --mem=96G \
#        -t 03:00:00 -J mapsim-calt -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/calib_theta.sh
set -o pipefail
NOISE=${NOISE:-forecast}
if [ "$NOISE" = forecast ]; then VAR=calibtf41; MAPS=maps_npe_forecast; else VAR=calibt41; MAPS=maps_npe; fi
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
step "node"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; cat $CODE/SNAPSHOT
step "1. catalogue tests"
PYTHONPATH=$CODE $PY -m pytest -q -p no:cacheprovider -s tests/test_sim_catalogue.py 2>&1 | quiet | grep -v "^\s*$" | tail -16; check ${PIPESTATUS[0]}
step "2. the reworked per-halo table against the mocks' data vectors (2 mocks)"
PYTHONPATH=$CODE $PY - <<PYEOF 2>&1 | quiet; check ${PIPESTATUS[0]}
import sys, numpy as np
sys.path.insert(0, "benchmarks/mapsim")
import qcal
H, T = qcal.table("$NOISE", 2, log=lambda m: None)
x = np.load("$MAPSIM_OUT/$MAPS/truth.npz")["x"][:2, :20].reshape(2, 4, 5)
Q = np.logspace(np.log10(5.0), np.log10(40.0), 5); Z = np.linspace(0.01, 1.5, 6)
for i in range(2):
    h = H[H["mock"] == i]; q, z = h["q_obs"], h["z"]
    m = np.isfinite(q) & (q > Q[0]) & (q < Q[-1]) & (z > Z[0]) & (z < Z[-1])
    c = np.zeros((4, 5)); np.add.at(c, (np.searchsorted(Q[1:-1], q[m]), np.searchsorted(Z[1:-1], z[m])), 1)
    print(f"mock {i}: one-per-halo counts {int(c.sum())} against the mock's {int(x[i].sum())}; max bin difference {int(np.abs(c - x[i]).max())}")
    assert np.abs(c - x[i]).max() <= 2, "the per-halo table does not reproduce the mock"
PYEOF
step "3. calibration at 25 points of the prior"
PYTHONPATH=$CODE $PY benchmarks/mapsim/qcal_grid.py run --noise $NOISE 2>&1 | quiet; check ${PIPESTATUS[0]}
PYTHONPATH=$CODE $PY benchmarks/mapsim/qcal_grid.py fit --noise $NOISE 2>&1 | quiet; check ${PIPESTATUS[0]}
for s in truth heldout train; do
  step "4. catalogue model with the calibration following the parameters ($VAR): $s"
  PYTHONPATH=$CODE $PY benchmarks/mapsim/catalogue_trainset.py --variant $VAR --set $s 2>&1 | quiet | grep -v "/10000 done\|qcal_theta\|spectra \["; check ${PIPESTATUS[0]}
done
step "5. its posterior, applied to the map-level mocks and prior draws"
$PYSBI benchmarks/mapsim/catalogue_npe.py --data $MAPSIM_OUT/catalogue_npe/$VAR --prior maps --tag t2calt --samples 4000 \
    --apply maps=$MAPSIM_OUT/$MAPS/truth.npz --apply mapheld=$MAPSIM_OUT/$MAPS/heldout.npz 2>&1 | quiet; check ${PIPESTATUS[0]}
step "done"
