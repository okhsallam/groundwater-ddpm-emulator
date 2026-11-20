#!/bin/bash -l
#PBS -N ceser-f15-fold4
#PBS -A <your_allocation>
#PBS -l select=1:ngpus=8
#PBS -j oe
#PBS -l walltime=14:00:00

# Fold 4: train [2004, 2005, 2006], test WY2003.
# Everything the rollout is conditioned on is static data, NLDAS-2 forcing, or
# the model's own output. Results and rollout live together under Results/$RUN.

cd "${PBS_O_WORKDIR:-$(dirname "$0")}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
PY=${PYTHON:-python3}
ACC=${ACCELERATE:-accelerate}

RUN="groundwater_f15_200ep_fold4"
CFG="config_f15_fold4.yaml"
CKPT="Results/$RUN/checkpoints/final_model.pt"
OUT="Results/$RUN/rollout"

echo "=== [1/4] training (train [2004, 2005, 2006], test WY2003, no val) ==="
$ACC launch --num_processes 8 train.py --config "$CFG" --name "$RUN"

echo "=== [2/4] test-set evaluation (single call, by lead day) ==="
$PY evaluate_test.py --config "$CFG" --checkpoint "$CKPT" \
    --out-dir "Results/$RUN/test_eval" --n-eval 50

echo "=== [3/4] ensemble rollout, 16 members, bounds in-loop ==="
$PY run_ensemble_rollout.py --config "$CFG" --checkpoint "$CKPT" \
    --out-dir "$OUT" --n-members 16

echo "=== [4/4] verification and figures ==="
$PY analyze_ensemble.py --rollout-dir "$OUT" --out-dir "$OUT/analysis" \
    --label "fold 4, test WY2003, 15-day window, 16 members"
$PY postprocess_ensemble.py --rollout-dir "$OUT" --out-dir "$OUT/analysis" \
    --label "fold 4, test WY2003"
$PY plot_rmse_evolution_clean.py --rollout-dir "$OUT" --out-dir "$OUT/analysis"

echo "=== PIPELINE DONE: fold4 ==="
exit 0
