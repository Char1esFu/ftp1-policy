#!/bin/bash
#SBATCH --job-name=ftp1_convert
#SBATCH --account=g107ea
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/convert_%j.out
#SBATCH --error=slurm/logs/convert_%j.out
# UniVTAC HDF5 -> FTP-1 zarr for each released task. Episodes 0-49 are the
# paper's "50 official demonstrations" (frame counts match the release's
# dataset_stats); 50-59 are converted too and held out as a validation split by
# norm_stats.sbatch, so training data stays identical to the release.
# CPU-only work; alex has no CPU partition, so it borrows one A40.
#
#   cd ~/Projects/ftp1-policy && sbatch slurm/convert_univtac.sbatch [task ...]
#
# Output: $UNIVTAC_ZARR/<short_name>/<task>_{head,all}.zarr

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TASKS=("$@")
[ "${#TASKS[@]}" -gt 0 ] || TASKS=("${FTP1_UNIVTAC_TASKS[@]}")
EPISODES_PER_TASK="${EPISODES_PER_TASK:-60}"   # 50 train + 10 held-out val

# parse_data_univtac.py expects <base>/<task>/demo/hdf5/*.hdf5 and derives the
# training prompt and camera set from <task>, so link the raw clean/ dirs under
# full task names instead of touching the shared raw data.
LINKS="$FTP1_CACHE/univtac_hdf5_links"
for t in "${TASKS[@]}"; do
    mkdir -p "$LINKS/$t/demo"
    ln -sfn "$UNIVTAC_RAW/$t/clean" "$LINKS/$t/demo/hdf5"
done

pids=()
for t in "${TASKS[@]}"; do
    out="$UNIVTAC_ZARR/$(ftp1_short_name "$t")"
    mkdir -p "$out"
    echo "[convert] $t -> $out"
    BASE_DIR="$LINKS" SAVE_DIR="$out" TASK_LIST="$t" EPISODES_PER_TASK="$EPISODES_PER_TASK" \
        ftp1_run bash data_processing/parse_data_scripts/parse_data_univtac.sh > "slurm/logs/convert_${SLURM_JOB_ID}_$t.log" 2>&1 &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done

for t in "${TASKS[@]}"; do
    echo "[convert] $t: $(grep -h 'wrote' "slurm/logs/convert_${SLURM_JOB_ID}_$t.log" || echo 'FAILED, see log')"
done
exit $status
