#!/bin/bash -l
# The bias test at 40% of the sky (168 tiles, forecast noise): ensembles of 5 networks (seeds 11-15, 2 threads each,
# 8 at a time) for the map-level posterior and three catalogue posteriors (published style, calibrated at one point,
# calibrated through the noise level), each applied to the 100 map-level mocks and the 500 map-level prior draws;
# then npe_ensemble.py per model. Submit (with the user's approval):
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p genx -C rome -c 16 --mem=48G -t 04:00:00 \
#        -J mapsim-n40 -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/npe40.sh
set -o pipefail
TILES=${TILES:-168}
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/mnt/home/bbolliet/venvs/sbi/bin/python
MAPS=$MAPSIM_OUT/maps_npe_forecast_t$TILES
cd $CODE
echo "=== $(date +%T) node"; hostname; cat $CODE/SNAPSHOT
declare -A DATA=([map]=$MAPS [plain]=$MAPSIM_OUT/catalogue_npe/forecast41_t$TILES [fixed]=$MAPSIM_OUT/catalogue_npe/calibf41_t$TILES [noise]=$MAPSIM_OUT/catalogue_npe/calibcf41_t$TILES)
mkdir -p $MAPS/robust40
run() {  # model seed
  $PY benchmarks/mapsim/catalogue_npe.py --data ${DATA[$1]} --prior maps --tag ${1}_e$2 --seed $2 --samples 4000 \
      --apply mocks=$MAPS/truth.npz --apply held=$MAPS/heldout.npz > $MAPS/robust40/${1}_e$2.log 2>&1
  echo "$(date +%T) $1 seed $2: exit $?"
}
jobs_list=()
for m in map noise fixed plain; do for s in 11 12 13 14 15; do jobs_list+=("$m $s"); done; done
for j in "${jobs_list[@]}"; do
  while [ $(jobs -rp | wc -l) -ge 8 ]; do sleep 20; done
  run $j &
done
wait
for m in map noise fixed plain; do
  echo; echo "=== $(date +%T) ensemble: $m"
  $PY benchmarks/mapsim/npe_ensemble.py ${DATA[$m]} --tags ${m}_e11,${m}_e12,${m}_e13,${m}_e14,${m}_e15 --sets mocks,held
done
echo "=== $(date +%T) done"
