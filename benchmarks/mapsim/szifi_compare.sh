#!/bin/bash -l
# SZiFi against our finder on MSS2 cut-outs with the simulator's sky:
#   1. mss2_szifi_maps.py     the maps and our finder's catalogues (classy_szlite environment, JAX on CPU)
#   2. mss2_szifi_run.py      SZiFi itself in three configurations (SZiFi's environment, one thread per run)
#   3. mss2_szifi_compare.py  the comparison (summary.txt, summary.json)
# Submit (with the user's approval) from a frozen snapshot, e.g.:
#   CODE=/mnt/home/bbolliet/mapsim-runs/<sha> sbatch -p genx -c 16 --mem=64G -t 02:00:00 -J mapsim-szifi \
#        -o /mnt/ceph/users/bbolliet/mapsim/m0/logs/%x-%j.out $CODE/benchmarks/mapsim/szifi_compare.sh
# A step that fails stops the job. Outputs of an earlier job are moved aside first (never mixed in).
set -o pipefail
CODE=${CODE:-/mnt/home/bbolliet/classy_szlite}
WORKERS=${WORKERS:-12}
export MAPSIM_OUT=/mnt/ceph/users/bbolliet/mapsim
export MAPSIM_REFDATA=/mnt/ceph/users/bbolliet/sz-poisson-blending
export SZIFI_DIR=/mnt/ceph/users/bbolliet/szifi
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/mnt/home/bbolliet/.cache/jax_mapsim
PY_SIM=/mnt/home/bbolliet/venvs/classy_szlite/bin/python
PY_SZ=/mnt/home/bbolliet/venvs/szifi/bin/python
DEST=$MAPSIM_OUT/mss2/szifi_compare
cd $CODE
step() { echo; echo "=== $(date +%T) $*"; }
quiet() { grep --line-buffered -v -i warn; }
check() { [ "$1" -eq 0 ] || { echo "step failed (exit $1); stopping"; exit "$1"; }; }
step "node"; hostname; nproc; free -g | head -2
cat $CODE/SNAPSHOT 2>/dev/null || git log --oneline -1
(cd $SZIFI_DIR && git log --oneline -1 && git status --short | head -3)
if [ -d $DEST ]; then OLD=$DEST.old.$(date +%Y%m%d-%H%M%S); mv $DEST $OLD; echo "earlier outputs moved to $OLD"; fi
step "1. the maps and our finder's catalogues"
PYTHONPATH=$CODE $PY_SIM benchmarks/mapsim/mss2_szifi_maps.py 2>&1 | quiet; check ${PIPESTATUS[0]}
step "2. SZiFi ($WORKERS workers)"
$PY_SZ -I -B benchmarks/mapsim/mss2_szifi_run.py --workers $WORKERS 2>&1 | quiet; check ${PIPESTATUS[0]}
step "3. comparison"
$PY_SZ -I -B benchmarks/mapsim/mss2_szifi_compare.py 2>&1 | quiet; rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] || echo "the comparison's own checks failed (exit $rc): see its first lines"
step "SZiFi clone unchanged?"
(cd $SZIFI_DIR && git status --short | head -3)
step "done"
exit $rc
