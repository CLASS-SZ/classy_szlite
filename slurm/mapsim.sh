#!/bin/bash -l
# Run one `mapsim` step as a Slurm job. Configure with environment variables, pass the step as arguments:
#
#   export MAPSIM_OUT=/path/to/outputs                 # where every step reads and writes
#   export MAPSIM_PYTHON=/path/to/venv/bin/python      # a Python with classy_szlite (and the [sbi] extra for `train`)
#   # GPU steps (tile, maps, calibrate point|grid, catalogues):
#   sbatch -p <gpu partition> --gpus=1 --cpus-per-gpu=16 --mem=96G -t 04:00:00 slurm/mapsim.sh maps --set train --tiles 41
#   # CPU steps (calibrate fit-noise|fit-theta, train, compare):
#   sbatch -p <cpu partition> -c 16 --mem=32G -t 02:00:00 slurm/mapsim.sh train --data maps --seed 11
#
# Every step resumes where it stopped, so a job that hits its time limit can simply be submitted again.
# Optional: CLASSY_SZLITE_DATA_DIR (emulator weights), CLASSY_SZLITE_CACHE (paint tables),
# JAX_COMPILATION_CACHE_DIR (compiled programs, reused across jobs), MAPSIM_CMB_TABLE (a CAMB CMB table).
set -euo pipefail
: "${MAPSIM_OUT:?set MAPSIM_OUT}"
PY=${MAPSIM_PYTHON:-python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-4}}}
echo "=== $(date +%T) $(hostname): mapsim $*"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -m classy_szlite.sim "$@" --out "$MAPSIM_OUT"
echo "=== $(date +%T) done"
