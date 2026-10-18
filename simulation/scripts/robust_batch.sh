#!/usr/bin/env bash
# Robustness batch (5 eval workers, sequential). Each robust_report call skips conditions it already
# has, so the batch can be re-run after an interruption.  bash simulation/scripts/robust_batch.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniconda3/envs/humanego-sim/bin/python}
R="$PY scripts/robust_report.py --workers 5"
F='perception.mask_frames=5'
# A. mitigation on the development seeds (Trossen)
$R --seed 7777 --n 32 --name trossen_dev7777 --conditions clean masks depth calib latency bread_size layout table_height lighting distractors all
$R --seed 7777 --n 32 --name trossen_dev7777_fuse5 --set "$F" --conditions masks all
$R --seed 8888 --n 32 --name trossen_dev8888 --conditions clean masks all
$R --seed 8888 --n 32 --name trossen_dev8888_fuse5 --set "$F" --conditions masks all
# B. final report, reserved scene set 9001: the key conditions for every robot first
for r in trossen_wxai franka_panda ur5e; do
  $R --robot $r --seed 9001 --n 100 --final --log --name ${r}_9001 --conditions clean masks dinosam all
done
# C. multi-frame mask fusion on the reserved set (rows of the final table; selection was on A)
for r in trossen_wxai franka_panda ur5e; do
  $R --robot $r --seed 9001 --n 100 --final --name ${r}_9001_fuse5 --set "$F" --conditions masks all
done
# D. the remaining single perturbations
for r in trossen_wxai franka_panda ur5e; do
  $R --robot $r --seed 9001 --n 100 --final --log --name ${r}_9001 \
     --conditions depth calib latency bread_size layout table_height lighting distractors
done
echo "[batch] done"
