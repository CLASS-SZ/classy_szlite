#!/bin/bash -l
# M6b GPU step on rusty: the MSS2-calibrated noise path on the GPU.
# Submit (with the user's approval) from a frozen snapshot, e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p gpu --gpus=1 -C rtxblackwell --cpus-per-gpu=16 \
#        --mem=64G -t 01:00:00 -J mapsim-noise -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out \
#        $CODE/benchmarks/mapsim/noise_gpu.sh
# Every step runs even if an earlier one fails.
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
source /mnt/home/bbolliet/venvs/classy_szlite/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
cd $CODE
export PYTHONPATH=$CODE
SP="python benchmarks/mapsim/spike.py"
step() { echo; echo "=== $(date +%T) $*"; }
step "node"; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cat $CODE/SNAPSHOT 2>/dev/null || git log --oneline -1
python -c "import jax, classy_szlite; print(jax.__version__, jax.devices(), classy_szlite.__file__)"
step "1. the noise tests on the GPU backend (fast and slow; deterministic ops for the integer batch test)"
XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m pytest -q -p no:cacheprovider -m "not gpu" tests/test_sim_noise.py 2>&1 | tail -8
step "2. GPU against CPU: 4 tiles, one MSS2 spectrum per tile, depths 1.0, 1.3, 0.8"
$SP tiles --batch 4 --noise mss2 --mixed-depth --tag noise_gpu 2>&1 | grep -v -i warn | tail -5
JAX_PLATFORMS=cpu $SP tiles --batch 4 --noise mss2 --mixed-depth --tag noise_cpu 2>&1 | grep -v -i warn | tail -5
$SP compare --tags noise_gpu noise_cpu 2>&1 | grep -v -i warn
step "3. batch invariance with MSS2 noise per tile, deterministic ops"
XLA_FLAGS=--xla_gpu_deterministic_ops=true $SP invariance --batch 1 4 16 --noise mss2 --mixed-depth 2>&1 | grep -v -i warn | tail -12
step "4. throughput and memory: the 2019 forecast noise against the MSS2 noise"
$SP run --batch 1 8 16 --reps 3 --tag rtx_noise_forecast 2>&1 | grep -v -i warn | grep -E "B=|peak|overflow"
$SP run --batch 1 8 16 --reps 3 --tag rtx_noise_mss2 --noise mss2 2>&1 | grep -v -i warn | grep -E "B=|peak|overflow|noise:"
step "5. peak buffer at the deepest MSS2 tile: the fiducial point and the prior box's heaviest corner"
$SP tiles --batch 3 --noise mss2 --deepest --tag deepest_fid 2>&1 | grep -v -i warn | tail -4
$SP tiles --batch 3 --noise mss2 --deepest --theta corner --tag deepest_corner 2>&1 | grep -v -i warn | tail -4
step "done"
