#!/bin/bash -l
#PBS -N ceser-f15-fold1
#PBS -A <your_allocation>
#PBS -l select=1:ngpus=8
#PBS -j oe
#PBS -l walltime=14:00:00

# Fold 1: train [2003, 2004, 2005], test WY2006.
#
# Reconstructed 2026-10-08. The fold-1 results in
# Results/groundwater_f15_200ep_fold1 were originally produced by
# pipeline_modified_f15.sh under the run name groundwater_modified_f15_200ep_8gpu
# and renamed afterwards, so no fold-1 pipeline existed. This reproduces that run
# from config_f15_fold1.yaml (itself the renamed config_modified_f15.yaml) and is
# identical in form to folds 2-4.
#
# Everything the rollout is conditioned on is static data, NLDAS-2 forcing, or
# the model's own output. Results and rollout live together under Results/$RUN.

cd "${PBS_O_WORKDIR:-$(dirname "$0")}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
PY=${PYTHON:-python3}
ACC=${ACCELERATE:-accelerate}

RUN="groundwater_f15_200ep_fold1"
CFG="config_f15_fold1.yaml"
CKPT="Results/$RUN/checkpoints/final_model.pt"
OUT="Results/$RUN/rollout"

echo "=== [1/4] training (train [2003, 2004, 2005], test WY2006, no val) ==="
$ACC launch --num_processes 8 train.py --config "$CFG" --name "$RUN"

echo "=== [2/4] test-set evaluation (single call, by lead day) ==="
$PY evaluate_test.py --config "$CFG" --checkpoint "$CKPT" \
    --out-dir "Results/$RUN/test_eval" --n-eval 50

echo "=== [3/4] ensemble rollout, 16 members, bounds in-loop ==="
$PY run_ensemble_rollout.py --config "$CFG" --checkpoint "$CKPT" \
    --out-dir "$OUT" --n-members 16

echo "=== [4/4] verification and figures ==="
$PY analyze_ensemble.py --rollout-dir "$OUT" --out-dir "$OUT/analysis" \
    --label "fold 1, test WY2006, 15-day window, 16 members"
$PY postprocess_ensemble.py --rollout-dir "$OUT" --out-dir "$OUT/analysis" \
    --label "fold 1, test WY2006"
$PY plot_rmse_evolution_clean.py --rollout-dir "$OUT" --out-dir "$OUT/analysis"

echo "=== PIPELINE DONE: fold1 ==="
exit 0
