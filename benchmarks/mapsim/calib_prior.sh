#!/bin/bash -l
# Does a calibration made at one point hold across the prior? The catalogue-level posteriors (plain, survey41;
# calibrated at the fiducial point, calib41) and the map-level posterior, each retrained (same seed) and applied to
# the 500 map-level held-out prior draws: coverage and bias against sigma8. Submit (with the user's approval):
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p genx -C rome -c 16 --mem=32G -t 01:00:00 \
#        -J mapsim-calprior -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/calib_prior.sh
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=/mnt/home/bbolliet/venvs/sbi/bin/python
cd $CODE
quiet() { tr '\r' '\n' | grep --line-buffered -v -i "warn\|Epochs trained:\|Drawing .* samples\|it/s\]\|s/it\]\|sample_batched"; }
echo "=== $(date +%T) node"; hostname; cat $CODE/SNAPSHOT
rc=0
for v in calib41 survey41 maps_npe; do
  d=$MAPSIM_OUT/catalogue_npe/$v; [ $v = maps_npe ] && d=$MAPSIM_OUT/maps_npe
  echo; echo "=== $(date +%T) $v"
  $PY benchmarks/mapsim/catalogue_npe.py --data $d --prior maps --tag prior_$v --samples 4000 \
      --apply mapheld=$MAPSIM_OUT/maps_npe/heldout.npz --apply maps=$MAPSIM_OUT/maps_npe/truth.npz 2>&1 | quiet \
      | sed -n '/^3\. applied/,$p'
  [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
done
echo "=== $(date +%T) done ($rc)"; exit $rc
