#!/bin/bash
#SBATCH --job-name=ftp1_probe
#SBATCH --account=g107ea
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/probe_%j.out
#SBATCH --error=slurm/logs/probe_%j.out
# Does the paper's per-GPU batch (8) fit on one 48 GB A40, and how fast is a step?
# Runs train_univtac.sbatch's exact config for a few steps on one GPU, records
# peak GPU memory, then deletes the throwaway checkpoint.
#
#   cd ~/Projects/ftp1-policy && sbatch slurm/probe_batch.sbatch [task] [per_gpu_batch] [grad_ckpt]
#   e.g. sbatch slurm/probe_batch.sbatch insert_tube 8 false

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TASK="${1:-insert_tube}"
export GLOBAL_BATCH="${2:-8}"
export GRAD_CKPT="${3:-false}"
export NUM_TRAIN_STEPS="${PROBE_STEPS:-60}"
export SAVE_INTERVAL=1000000
export EXP_SUFFIX="_probe_b${GLOBAL_BATCH}_gc${GRAD_CKPT}_${SLURM_JOB_ID}"
export WANDB_MODE=disabled   # throwaway run; step time and memory go to this log
export VAL_INTERVAL=1000000

MEMLOG="slurm/logs/probe_${SLURM_JOB_ID}_mem.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 2 > "$MEMLOG" &
SMI=$!

set +e
bash slurm/train_univtac.sbatch "$TASK"
rc=$?
set -e
kill "$SMI" 2>/dev/null || true

peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}' "$MEMLOG")
echo "=== probe $TASK per_gpu_batch=$GLOBAL_BATCH grad_ckpt=$GRAD_CKPT exit=$rc peak_gpu_mem=${peak} MiB of $(nvidia-smi --query-gpu=memory.total --format=csv,noheader) ==="
rm -rf "$CHECKPOINT_BASE_DIR/ftp1/UniVTAC_$(ftp1_short_name "$TASK")_expert_gsmall_ftp1${EXP_SUFFIX}"
exit $rc
