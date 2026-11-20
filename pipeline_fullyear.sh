#!/bin/bash -l
#PBS -N ceser-fullyear-data
#PBS -A <your_allocation>
#PBS -l select=1:ncpus=8
#PBS -l walltime=24:00:00
#PBS -j oe

# Stage 1 of the full-year seasonal experiment: acquire and prepare the forcing.
#
# Runs on a compute node, which was verified to have outbound internet (the CDS
# answers 401 rather than timing out, and Earthdata answers 200). Putting the
# download in PBS rather than on the login node means it survives a lost
# connection AND a login-node restart, and it is restartable: every step skips
# work already on disk, so resubmitting after any failure resumes rather than
# repeats.
#
# Inference is a SEPARATE job (pipeline_fullyear_infer.sh) submitted with a
# dependency on this one, because it needs 8 GPUs for ~13 h and this stage needs
# none. Holding GPUs idle through a multi-hour download would waste them and
# risk the 24 h walltime.

cd "${PBS_O_WORKDIR:-$(dirname "$0")}"
PY=${PYTHON:-python3}

echo "=== node $(hostname), started $(date) ==="

# ---- 1. the four May re-initialisations -----------------------------------
echo "--- downloading May initialisations ---"
$PY -u seas5_fullyear_runner.py --days 215
n=$(ls data/seas5/raw_fullyear/seas5_2*05_*.nc 2>/dev/null | wc -l)
echo "May initialisations on disk: $n of 4"
if [ "$n" -lt 4 ]; then
    echo "INCOMPLETE: resubmit this script to resume the remaining downloads."
    exit 1
fi

# ---- 2. coarse-grid forcing for every initialisation ----------------------
# The October files are already prepared; this adds the May ones. The stage is
# idempotent and skips an initialisation whose coarse file already exists.
echo "--- preparing coarse forcing ---"
for f in data/seas5/raw_fullyear/seas5_2*05_*.nc; do
    tag=$(basename "$f" | sed 's/seas5_\([0-9]*\)_.*/\1/')
    if [ -f "data/seas5/forcing_coarse_seas5_${tag}.nc" ]; then
        echo "  $tag: already prepared"; continue
    fi
    cp -n "$f" "data/seas5/raw/seas5_${tag}.nc"
    $PY -u prepare_seas5_domain.py --stages seas5_raw || exit 1
done

echo "--- verifying the spliced year is continuous ---"
$PY -u verify_fullyear_forcing.py || exit 1

echo "=== finished $(date) ==="
