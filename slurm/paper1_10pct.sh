#!/bin/bash
# The 10%-sky bias test (41 tiles, forecast noise) as a chain of Slurm jobs. Edit the partitions, then run
# this script from the repository root (it only submits). Total: about 2 GPU hours and 2 CPU hours.
set -euo pipefail
GPU="-p gpu --gpus=1 --cpus-per-gpu=16 --mem=96G"
CPU="-p genx -c 16 --mem=32G"
sub() { sbatch --parsable "$@"; }
J1=$(sub $GPU -t 03:00:00 -J ms-maps slurm/mapsim.sh maps --set truth)
J2=$(sub $GPU -t 03:00:00 -J ms-maps --dependency=afterok:$J1 slurm/mapsim.sh maps --set heldout)
J3=$(sub $GPU -t 04:00:00 -J ms-maps --dependency=afterok:$J2 slurm/mapsim.sh maps --set train)
J4=$(sub $GPU -t 01:00:00 -J ms-cal slurm/mapsim.sh calibrate point)
J5=$(sub $GPU -t 02:00:00 -J ms-cal --dependency=afterok:$J4 slurm/mapsim.sh calibrate grid)
J6=$(sub $CPU -t 00:30:00 -J ms-cal --dependency=afterok:$J5 slurm/mapsim.sh calibrate fit-noise)
CAT=""
for m in published calibrated-point calibrated-noise; do
  dep=$([ $m = published ] && echo "" || ([ $m = calibrated-point ] && echo "--dependency=afterok:$J4" || echo "--dependency=afterok:$J6"))
  for s in truth heldout train; do
    J=$(sub $GPU -t 01:00:00 -J ms-cat $dep slurm/mapsim.sh catalogues --model $m --set $s); CAT="$CAT:$J"
  done
done
TR=""
for d in maps published calibrated-point calibrated-noise; do
  for seed in 11 12 13 14 15; do
    J=$(sub $CPU -t 03:00:00 -J ms-train --dependency=afterok:$J3$CAT slurm/mapsim.sh train --data $d --seed $seed); TR="$TR:$J"
  done
done
sub $CPU -t 00:20:00 -J ms-compare --dependency=afterok${TR} slurm/mapsim.sh compare
