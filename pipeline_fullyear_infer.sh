#!/bin/bash -l
#PBS -N ceser-fullyear-infer
#PBS -A <your_allocation>
#PBS -l select=1:ngpus=8
#PBS -l walltime=24:00:00
#PBS -j oe

# Stage 2: the 360-day seasonal rollout, four folds, forced by the spliced
# October + May SEAS5 forecast.
#
# Submitted with -W depend=afterok on the data job, so it starts only once the
# forcing exists and has been verified continuous. If the data job fails, this
# never runs and no GPU time is wasted.
#
# Restartable: run_seas5_rollout.py skips a member whose output file already
# exists, so resubmitting after a walltime kill resumes from where it stopped.

cd "${PBS_O_WORKDIR:-$(dirname "$0")}"
PY=${PYTHON:-python3}
declare -A INIT=( [1]=200510 [2]=200410 [3]=200310 [4]=200210 )
ARM=Dyear                      # raw SEAS5, spliced to a full water year
HOPS=24                        # 24 x 15 d = 360 d, the same window as NLDAS

echo "=== node $(hostname), started $(date) ==="
$PY -u verify_fullyear_forcing.py || exit 1

# Fail loudly if no member is produced. The first attempt verified the forcing,
# then every member died in seconds on a missing coarse file, and the job still
# reported success because nothing checked the output existed.
COUNT_BEFORE=$(ls ddpm_model/Results/seas5/arm${ARM}_fold*/member_*.npz 2>/dev/null | wc -l)

GPU=0; PIDS=()
for f in 1 2 3 4; do
    OUT="ddpm_model/Results/seas5/arm${ARM}_fold$f"
    mkdir -p "$OUT"
    for m in 0 1 2 3 4 5 6 7 8 9; do
        printf -v MF "%s/member_%03d.npz" "$OUT" "$m"
        [ -f "$MF" ] && continue
        ( cd ddpm_model && CUDA_VISIBLE_DEVICES=$GPU $PY run_seas5_rollout.py \
            --config "config_f15_fold$f.yaml" \
            --checkpoint "Results/groundwater_f15_200ep_fold$f/checkpoints/final_model.pt" \
            --arm "$ARM" --init "${INIT[$f]}" --out-dir "Results/seas5/arm${ARM}_fold$f" \
            --n-diff 4 --members "$m" --max-hops $HOPS ) \
            > "$OUT/m$m.log" 2>&1 &
        PIDS+=($!); GPU=$(( (GPU+1) % 8 ))
        if [ ${#PIDS[@]} -ge 8 ]; then
            for p in "${PIDS[@]}"; do wait $p; done; PIDS=()
        fi
    done
done
for p in "${PIDS[@]}"; do wait $p; done
echo "=== finished $(date) ==="
TOTAL=0
for f in 1 2 3 4; do
    k=$(ls ddpm_model/Results/seas5/arm${ARM}_fold$f/member_*.npz 2>/dev/null | wc -l)
    echo "fold$f: $k members"; TOTAL=$((TOTAL+k))
done
echo "total members: $TOTAL of 40"
if [ "$TOTAL" -le "$COUNT_BEFORE" ]; then
    echo "FAILED: no new member was produced. First member log:"
    head -5 ddpm_model/Results/seas5/arm${ARM}_fold1/m0.log 2>/dev/null
    exit 1
fi
[ "$TOTAL" -lt 40 ] && echo "PARTIAL: resubmit to resume the remaining members."
exit 0
