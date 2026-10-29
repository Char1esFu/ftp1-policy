#!/bin/bash
#SBATCH --job-name=ftp1_norm
#SBATCH --account=g107ea
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/norm_%j.out
#SBATCH --error=slurm/logs/norm_%j.out
# Per-task normalization stats, same flags as the authors'
# scripts_exp_zarr/univtac_example_real_machine/batch_compute_norm_stats_univtac_all_tasks.sh
# (full-data zscore, channel_wise, mix actions, contact thresholds).
#
#   cd ~/Projects/ftp1-policy && sbatch slurm/norm_stats.sbatch [task ...]
#
# Writes $ASSETS_BASE_DIR/ftp1/UniVTAC_<short>_zscore_mix_ftp1/ and the dataset
# config slurm/configs/dataset_univtac_<short>.json that training reuses.
#
# Train/val split: the release trained on episodes 0-49 with an empty val split
# (val_ratio 0.01 of 50 rounds to 0), so it logged no validation at all. We
# pre-write train_val_split.json as train 0-49 / val 50-59 before the stats run;
# the loader keeps an existing split, norm stats use the train split only, so
# stats and training data match the release and 50-59 give a held-out val curve.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TASKS=("$@")
[ "${#TASKS[@]}" -gt 0 ] || TASKS=("${FTP1_UNIVTAC_TASKS[@]}")
mkdir -p slurm/configs

for t in "${TASKS[@]}"; do
    short="$(ftp1_short_name "$t")"
    task_dir="$UNIVTAC_ZARR/$short"
    if ! ls -d "$task_dir"/*.zarr >/dev/null 2>&1; then
        echo "[norm] $t: no zarr under $task_dir; run convert_univtac.sbatch first" >&2
        exit 1
    fi
    cfg="slurm/configs/dataset_univtac_${short}.json"
    cat > "$cfg" <<EOF
{
  "datasets": [
    {"name": "UniVTAC_${short}", "path": "${task_dir}", "use_trajectory_ratio": 1.0, "enabled": true}
  ],
  "default_use_trajectory_ratio": 1.0,
  "description": "Per-task UniVTAC config matching the ftp1_univtac_finetune release."
}
EOF
    repo_id="UniVTAC_${short}_zscore_mix_ftp1"
    split_dir="$ASSETS_BASE_DIR/ftp1/$repo_id/UniVTAC_${short}"
    mkdir -p "$split_dir"
    split_file="$split_dir/train_val_split.json"
    ftp1_run python - "$split_file" "$task_dir" <<'EOF'
import glob, json, sys, zarr
out, task_dir = sys.argv[1], sys.argv[2]
n = sum(zarr.open(p, mode="r")["meta/episode_ends"].shape[0] for p in glob.glob(f"{task_dir}/*.zarr"))
assert n >= 60, f"{task_dir} has {n} episodes, need 60 (50 train + 10 val)"
json.dump({
    "train_episode_idx": list(range(50)),
    "val_episode_idx": list(range(50, 60)),
    "_split_strategy_version": 4,  # dataset_zarr.SPLIT_STRATEGY_VERSION; a mismatch would regenerate the split
    "_split_strategy": "norm_stats_domain_name_unique_source_episode",
}, open(out, "w"), indent=2)
print(f"[norm] split {out}: train 0-49, val 50-59 of {n} episodes")
EOF
    echo "=== [norm] $t | domain UniVTAC_${short} | repo $repo_id ==="
    # The stats script wipes the repo dir first; keep the split written above.
    FTP1_PRESERVE_SPLIT=1 ftp1_run python scripts/zarr_compute_norm_stats.py ftp1 \
        --repo_id="$repo_id" \
        --data.repo-id="$repo_id" \
        --exp_name=default \
        --use_val_dataset \
        --create_train_val_split \
        --val_ratio=0.0 \
        --no-wandb_enabled \
        --checkpoint_base_dir="$CHECKPOINT_BASE_DIR" \
        --assets_base_dir="$ASSETS_BASE_DIR" \
        --dataset_config_path="$cfg" \
        --norm_sample_ratio=1.0 \
        --batch_size=32 \
        --norm_batch_size=64 \
        --norm_num_workers="${SLURM_CPUS_PER_TASK:-12}" \
        --action_down_sample_steps=1 \
        --norm_type=zscore \
        --norm_image_tactile_mode=channel_wise \
        --contact_detection_threshold_k=3.0 \
        --no-skip_contact_detection \
        --proprioception_pose_rep=relative \
        --action_pose_rep=relative \
        --proprioception_joint_rep=abs \
        --action_joint_rep=mix
    # Fail loudly if the loader replaced the split (it regenerates on any mismatch).
    ftp1_run python - "$split_file" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["train_episode_idx"] == list(range(50)) and d["val_episode_idx"] == list(range(50, 60)), \
    f"split was regenerated: train {len(d['train_episode_idx'])} / val {len(d['val_episode_idx'])} episodes"
print("[norm] split kept: train 0-49, val 50-59")
EOF
done
echo "[norm] done: ${TASKS[*]}"
