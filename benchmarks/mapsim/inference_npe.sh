#!/bin/bash -l
# The 10%-sky bias test (PLAN T1, T2) with catalogue_npe.py on CPU (sbi):
#   T2: the catalogue-level posterior of the 41-tile survey (catalogue_npe/survey41), validated on its own
#       mocks and held-out draws, then applied to the map-level mocks (maps_npe/truth.npz): the total bias
#       of a catalogue-level analysis of map-level data.
#   T1: the map-level posterior (maps_npe), validated on the same map-level mocks and the map-level held-out
#       draws: it must recover the truth with calibrated coverage.
# Submit (with the user's approval), e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p genx -C rome -c 16 --mem=32G -t 02:00:00 \
#        -J mapsim-bias -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/inference_npe.sh
# NOISE=forecast: the same with the public forecast noise (catalogue_npe/forecast41, maps_npe_forecast).
set -o pipefail
NOISE=${NOISE:-mss2}
if [ "$NOISE" = forecast ]; then CAT=forecast41; MAPS=maps_npe_forecast; else CAT=survey41; MAPS=maps_npe; fi
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/mnt/home/bbolliet/venvs/sbi/bin/python
cd $CODE
quiet() { tr '\r' '\n' | grep --line-buffered -v -i "warn\|Epochs trained:\|Drawing .* samples\|it/s\]\|s/it\]"; }
echo "=== $(date +%T) node"; hostname; cat $CODE/SNAPSHOT
echo "=== $(date +%T) T2: catalogue-level posterior of the 41-tile survey, applied to the map-level mocks"
$PY benchmarks/mapsim/catalogue_npe.py --data $MAPSIM_OUT/catalogue_npe/$CAT --prior maps --tag t2 \
    --apply maps=$MAPSIM_OUT/$MAPS/truth.npz 2>&1 | quiet; rc1=${PIPESTATUS[0]}
echo "=== $(date +%T) T1: map-level posterior"
$PY benchmarks/mapsim/catalogue_npe.py --data $MAPSIM_OUT/$MAPS --prior maps --tag t1 2>&1 | quiet; rc2=${PIPESTATUS[0]}
echo "=== $(date +%T) done (exits $rc1, $rc2)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
