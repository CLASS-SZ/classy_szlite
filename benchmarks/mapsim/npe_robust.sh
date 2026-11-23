#!/bin/bash -l
# Robustness of the map-level neural posterior (forecast noise by default): 5 networks with different seeds on
# the 10^4 training simulations, and one on 5,000, trained in parallel (2 threads each); each applied to the 100
# mocks and the 500 prior draws; then the ensemble (npe_ensemble.py). Submit (with the user's approval):
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p genx -C rome -c 16 --mem=32G -t 02:00:00 \
#        -J mapsim-robust -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/npe_robust.sh
set -o pipefail
NOISE=${NOISE:-forecast}
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
MAPS=$MAPSIM_OUT/maps_npe$([ "$NOISE" = forecast ] && echo _forecast)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/mnt/home/bbolliet/venvs/sbi/bin/python
cd $CODE
echo "=== $(date +%T) node"; hostname; cat $CODE/SNAPSHOT
mkdir -p $MAPS/robust
run() {  # tag seed n_train
  $PY benchmarks/mapsim/catalogue_npe.py --data $MAPS --prior maps --tag $1 --seed $2 --n-train $3 --samples 4000 \
      --apply mocks=$MAPS/truth.npz --apply held=$MAPS/heldout.npz > $MAPS/robust/$1.log 2>&1
  echo "$1 (seed $2, $3 simulations): exit $?"
}
for s in 0 1 2 3 4; do run e$s $((s + 11)) 10000 & done
run n5k 11 5000 &
wait
for t in e0 e1 e2 e3 e4 n5k; do echo "--- $t"; grep -a "converged\|TARP" $MAPS/robust/$t.log | tr '\r' '\n' | grep -v "^\s*$" | tail -2; done
echo "=== $(date +%T) ensemble"
$PY benchmarks/mapsim/npe_ensemble.py $MAPS --tags e0,e1,e2,e3,e4 --single n5k --sets mocks,held
echo "=== $(date +%T) done"
