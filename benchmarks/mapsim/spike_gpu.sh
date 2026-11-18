#!/bin/bash -l
# M0 GPU spike on rusty: parity, tests, painter check, throughput, invariance.
# Submit (with the user's approval) as, e.g.:
#   sbatch -p gpu --gpus=1 -C a100-80gb --cpus-per-gpu=16 --mem=128G -t 01:30:00 \
#          -J mapsim-spike -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out \
#          benchmarks/mapsim/spike_gpu.sh a100
# Every step runs even if an earlier one fails.
TAG=${1:-gpu}
source /mnt/home/bbolliet/venvs/classy_szlite/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
cd /mnt/home/bbolliet/classy_szlite
step() { echo; echo "=== $(date +%T) $*"; }
step "node"; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git log --oneline -1
python -c "import jax; print(jax.__version__, jax.devices())"
step "1. CPU vs GPU parity of the cosmology stage"
python -m pytest -q -p no:cacheprovider -m gpu tests -s 2>&1 | grep -v Warning | tail -6
step "2. default test suite on the GPU backend"
python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -4
step "3. painter vs exact synthesis"
python benchmarks/mapsim/spike.py check 2>&1 | grep -v -i warn | tail -9
step "4. throughput"
python benchmarks/mapsim/spike.py run --batch 1 2 4 8 16 --reps 3 --tag $TAG --trace 2>&1 | grep -v -i warn
step "5. batch invariance, deterministic ops"
XLA_FLAGS=--xla_gpu_deterministic_ops=true python benchmarks/mapsim/spike.py invariance --batch 1 4 16 2>&1 | grep -v -i warn | tail -24
step "6. throughput with deterministic ops"
XLA_FLAGS=--xla_gpu_deterministic_ops=true python benchmarks/mapsim/spike.py run --batch 8 --reps 2 --tag ${TAG}_det 2>&1 | grep -v -i warn | tail -9
step "done"
