#!/bin/bash -l
# Catalogue-level NPE (catalogue_npe.py): training as in the paper and its validation, on CPU with sbi.
# Submit (with the user's approval) from a frozen snapshot, e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p genx -C rome -c 16 --mem=16G -t 01:00:00 \
#        -J mapsim-npe -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/catalogue_npe.sh
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/mnt/home/bbolliet/venvs/sbi/bin/python
cd $CODE
echo "=== $(date +%T) node"; hostname; cat $CODE/SNAPSHOT
echo "=== $(date +%T) NPE, 10^4 simulations"
$PY benchmarks/mapsim/catalogue_npe.py --n-train 10000 --seed 0 2>&1 | grep --line-buffered -v -i "warn"
rc=${PIPESTATUS[0]}
echo "=== $(date +%T) done (exit $rc)"; exit $rc
