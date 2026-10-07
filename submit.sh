#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --job-name=rps
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load cuda12.6
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# sbatch submit.sh config.json runs 100000 31415
config=${1:?Usage: sbatch submit.sh CONFIG [OUTPUT_ROOT] [STEPS] [BASE_SEED]}
root=${2:-runs}
steps=${3:-100000}
seed=$(( ${4:-31415} + ${SLURM_ARRAY_TASK_ID:-0} ))
out="$root/job-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}-${SLURM_ARRAY_TASK_ID:-0}"
# Forward the early Slurm signal so Python can flush and save the final state.
trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
python -u batch.py run --config "$config" --out "$out" --steps "$steps" \
    --seed "$seed" --sample-every 100 --temporal &
child=$!
status=0
wait "$child" || status=$?
if kill -0 "$child" 2>/dev/null; then
    wait "$child" || status=$?
fi
exit "$status"
