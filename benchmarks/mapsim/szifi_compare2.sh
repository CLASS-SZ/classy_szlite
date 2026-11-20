#!/bin/bash -l
# Follow-up to szifi_compare.sh; needs its outputs in $MAPSIM_OUT/mss2/szifi_compare (left untouched):
#   0. tests of our filter on channel subsets (tests/test_sim_channels.py)
#   1. our finder on the maps with clusters again: all six channels (ours_rerun) and the four
#      high-frequency channels (ours_ch4); and the earlier snapshot's six-channel finder on the same
#      maps in this job (ours_ref): ours_rerun must equal it exactly
#   2. the same maps without the cluster sky (nosky/: MSS2 noise and CMB), our finder on them, and
#      its masked estimate there with the masks it used on the maps with clusters
#   3. SZiFi on the maps without clusters: set as our finder with four channels (match4) and the
#      colleagues' settings (coll), each also in one pass with its cluster masks (*_same)
#   4. ours_ch4 against SZiFi on the maps with clusters (summary_ch4), and each finder's noise after
#      masking against the noise without clusters (noise_summary)
# Submit (with the user's approval) from a frozen snapshot, on the CPU type of the first job (rome):
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p genx -C rome -c 16 --mem=64G -t 02:00:00 \
#        -J mapsim-szifi2 -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/szifi_compare2.sh
# A step that fails stops the job; this job's own earlier outputs are moved aside first.
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
REF_CODE=${REF_CODE:-/mnt/home/bbolliet/mapsim-runs/191b466}     # before the change to channel subsets
WORKERS=${WORKERS:-14}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export SZIFI_DIR=/mnt/ceph/users/bbolliet/szifi
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
PY_SIM=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
PY_SZ=/mnt/home/bbolliet/venvs/szifi/bin/python
DEST=$MAPSIM_OUT/mss2/szifi_compare
MAPS=benchmarks/mapsim/mss2_szifi_maps.py
cd $CODE
step() { echo; echo "=== $(date +%T) $*"; }
quiet() { grep --line-buffered -v -i warn; }
check() { [ "$1" -eq 0 ] || { echo "step failed (exit $1); stopping"; exit "$1"; }; }
step "node"; hostname; nproc; free -g | head -2; grep -m1 "model name" /proc/cpuinfo
cat $CODE/SNAPSHOT 2>/dev/null || git log --oneline -1
cat $REF_CODE/SNAPSHOT
(cd $SZIFI_DIR && git log --oneline -1 && git status --short | head -3)
[ -f $DEST/meta.json ] && [ -d $DEST/ours ] && [ -d $DEST/szifi/runs ] || { echo "the first comparison's outputs are missing"; exit 1; }
for d in nosky ours_ch4 ours_rerun ours_ref summary_ch4.txt summary_ch4.json noise_summary.txt noise_summary.json; do
  if [ -e $DEST/$d ]; then mv $DEST/$d $DEST/$d.old.$(date +%Y%m%d-%H%M%S); echo "moved $DEST/$d aside"; fi
done
step "0. our filter on channel subsets: tests"
PYTHONPATH=$CODE $PY_SIM -m pytest -q -p no:cacheprovider tests/test_sim_channels.py 2>&1 | quiet; check ${PIPESTATUS[0]}
step "1a. our finder on the maps with clusters: six channels again, and the four high-frequency channels"
PYTHONPATH=$CODE $PY_SIM $MAPS --root $DEST --no-write-maps \
    --variants "ours_rerun=0,1,2,3,4,5;ours_ch4=2,3,4,5" 2>&1 | quiet; check ${PIPESTATUS[0]}
step "1b. the earlier snapshot's six-channel finder on the same maps (reference)"
[ -f $REF_CODE/classy_szlite/sim/mmf.py ] || { echo "no reference package in $REF_CODE"; exit 1; }
got=$(cd / && PYTHONPATH=$REF_CODE $PY_SIM -c "import inspect, classy_szlite.sim.mmf as m; print(m.__file__, 'channels' in inspect.signature(m.filter_static).parameters)")
echo "reference package: $got"
[ "$got" = "$REF_CODE/classy_szlite/sim/mmf.py False" ] || { echo "step 1b would not use the reference package; stopping"; exit 1; }
PYTHONPATH=$REF_CODE $PY_SIM $MAPS --root $DEST --no-write-maps --variants "ours_ref=0,1,2,3,4,5" 2>&1 | quiet; check ${PIPESTATUS[0]}
step "2. the maps without the cluster sky, our finder on them, and with the masks from the maps with clusters"
PYTHONPATH=$CODE $PY_SIM $MAPS --root $DEST/nosky --no-clusters --mask-from $DEST \
    --variants "ours=0,1,2,3,4,5;ours_ch4=2,3,4,5" 2>&1 | quiet; check ${PIPESTATUS[0]}
step "3. SZiFi on the maps without clusters ($WORKERS workers)"
$PY_SZ -I -B benchmarks/mapsim/mss2_szifi_run.py --root $DEST/nosky --mask-from $DEST \
    --configs coll,coll_same,match4,match4_same --workers $WORKERS 2>&1 | quiet; check ${PIPESTATUS[0]}
step "4a. our finder with four channels against SZiFi, maps with clusters"
$PY_SZ -I -B benchmarks/mapsim/mss2_szifi_compare.py --root $DEST --ours ours_ch4 --configs match4,coll --tag ch4 \
    2>&1 | quiet; rc1=${PIPESTATUS[0]}
step "4b. noise after masking against the noise without clusters"
$PY_SZ -I -B benchmarks/mapsim/mss2_szifi_noise.py --root $DEST 2>&1 | quiet; rc2=${PIPESTATUS[0]}
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] || echo "a comparison's own checks failed (exits $rc1, $rc2): see its first lines"
step "SZiFi clone unchanged?"
(cd $SZIFI_DIR && git status --short | head -3)
step "done"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
