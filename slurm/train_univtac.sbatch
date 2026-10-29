#!/bin/bash
#SBATCH --job-name=ftp1_train
#SBATCH --account=g107ea
#SBATCH --partition=a40
# Paper App. D finetuning: 8 GPUs, global batch 64 (8 per GPU), 20k steps,
# lr 5e-5 -> 5e-6. One a40 node has exactly 8 cards.
# Fewer cards: sbatch --gres=gpu:a40:4 ... (per-GPU batch = GLOBAL_BATCH / #GPUs;
# add GRAD_CKPT=true if 16/GPU does not fit). CPUs (16) and RAM (60 GB) come with
# each A40 automatically, so no --cpus-per-task / --exclusive.
#SBATCH --gres=gpu:a40:8
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_%j.out
#SBATCH --error=slurm/logs/train_%j.out
# Reproduce one ftp1_univtac_finetune checkpoint (one model per task).
#
#   cd ~/Projects/ftp1-policy && sbatch slurm/train_univtac.sbatch <task>
#   e.g. sbatch slurm/train_univtac.sbatch insert_tube
#
# Resubmit the same command to continue: if the experiment already has a
# checkpoint the job resumes from the newest one instead of restarting.
#
# Overridable env vars:
#   NUM_TRAIN_STEPS (20000)  SAVE_INTERVAL (2000)  GLOBAL_BATCH (64)  VAL_INTERVAL (1000)
#   GRAD_CKPT (false)  EXP_SUFFIX ("")  -- appended to the experiment name
#
# Logs: wandb (offline under $WANDB_DIR, project $WANDB_PROJECT_NAME; upload with
# slurm/wandb_sync.sh). Every 100 steps: Trainning/{loss,learning_rate,epoch,
# time_per_step}, grad_norm, TrainDomainLoss/*. Every VAL_INTERVAL steps on the
# held-out episodes 50-59: Validation/* action errors of sampled chunks vs ground
# truth, plus ValidationFirstBatch/action_curves plots.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

TASK="${1:?usage: sbatch slurm/train_univtac.sbatch <task>}"
SHORT="$(ftp1_short_name "$TASK")"
REPO_ID="UniVTAC_${SHORT}_zscore_mix_ftp1"
EXP_NAME="UniVTAC_${SHORT}_expert_gsmall_ftp1${EXP_SUFFIX:-}"
DATASET_CFG="slurm/configs/dataset_univtac_${SHORT}.json"
EXP_DIR="$CHECKPOINT_BASE_DIR/ftp1/$EXP_NAME"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"   # only for resuming; the release saved once at 19999
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
GRAD_CKPT="${GRAD_CKPT:-false}"

[ -f "$DATASET_CFG" ] || { echo "missing $DATASET_CFG; run norm_stats.sbatch $TASK first" >&2; exit 1; }
[ -d "$ASSETS_BASE_DIR/ftp1/$REPO_ID" ] || { echo "missing norm stats $ASSETS_BASE_DIR/ftp1/$REPO_ID" >&2; exit 1; }
if [ $((GLOBAL_BATCH % NUM_GPUS)) -ne 0 ]; then
    echo "GLOBAL_BATCH=$GLOBAL_BATCH not divisible by $NUM_GPUS GPUs" >&2; exit 1
fi

# Each checkpoint is ~24 GB (8 GB weights + 16 GB AdamW) and the trainer never
# prunes, so keep only the newest complete step dir (plus the final one).
latest_step() {
    find "$EXP_DIR" -maxdepth 1 -mindepth 1 -type d -regex '.*/[0-9]+' -exec test -f {}/model.safetensors \; -print 2>/dev/null \
        | xargs -r -n1 basename | sort -n | tail -1
}
LATEST="$(latest_step || true)"
if [ -n "$LATEST" ]; then
    for d in "$EXP_DIR"/[0-9]*; do
        s="$(basename "$d")"
        if [ "$s" != "$LATEST" ] && [ "$s" != "$((NUM_TRAIN_STEPS - 1))" ]; then rm -rf "$d"; fi
    done
    rm -rf "$EXP_DIR"/tmp_*
fi

echo "=== FTP-1 $TASK | exp $EXP_NAME | $NUM_GPUS x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | batch $GLOBAL_BATCH | $(hostname) | $(date) ==="

ARGS=(
    ftp1
    --exp_name="$EXP_NAME"
    --repo_id="$REPO_ID"
    --data.repo-id="$REPO_ID"
    --checkpoint_base_dir="$CHECKPOINT_BASE_DIR"
    --assets_base_dir="$ASSETS_BASE_DIR"
    --dataset_config_path="$DATASET_CFG"
    --batch_size="$GLOBAL_BATCH"
    --num_train_steps="$NUM_TRAIN_STEPS"
    --save_interval="$SAVE_INTERVAL"
    --keep_period="$NUM_TRAIN_STEPS"
    --log_interval=100
    --val_interval="${VAL_INTERVAL:-1000}"
    --num_workers=12
    --val_num_workers=2
    --seed=42
    --use_val_dataset
    --create_train_val_split
    --val_ratio=0.01
    --no-check_norm_params_snapshot
    --wandb_enabled
    --project_name="$WANDB_PROJECT_NAME"
    --action_down_sample_steps=1
    --norm_type=zscore
    --norm_image_tactile_mode=channel_wise
    --pytorch_training_precision=bfloat16
    --use_torch_compile
    --model.tactile_expert_variant=gemma_small
    --model.use_tactile_input
    --model.state_input_mode=adarms
    --model.tactile_tokenizer_config.no_frozen_shared_chunk
    --proprioception_pose_rep=relative
    --action_pose_rep=relative
    --proprioception_joint_rep=abs
    --action_joint_rep=mix
    --lr_schedule.warmup_steps=500
    --lr_schedule.peak_lr=5e-5
    --lr_schedule.decay_steps="$NUM_TRAIN_STEPS"
    --lr_schedule.decay_lr=5e-6
    --lr_decay_till_end
    --optimizer.b1=0.9
    --optimizer.b2=0.95
    --optimizer.eps=1e-8
    --optimizer.weight_decay=1e-10
    --optimizer.clip_gradient_norm=1.0
    --ema_decay=0.99
)
if [ "$GRAD_CKPT" = "true" ]; then ARGS+=(--gradient_checkpointing_enable); else ARGS+=(--no-gradient_checkpointing_enable); fi

if [ -n "$LATEST" ]; then
    if [ "$LATEST" = "$((NUM_TRAIN_STEPS - 1))" ]; then
        echo "already finished: $EXP_DIR/$LATEST"; exit 0
    fi
    echo "resuming from $EXP_DIR/$LATEST"
    ARGS+=(--resume)
else
    echo "fresh run from $PRETRAIN_CKPT"
    # Clear the dir here instead of passing --overwrite: every rank runs the
    # trainer's overwrite rmtree, so a slow rank deletes the dir a fast rank has
    # already re-created and the others crash (FileNotFoundError).
    rm -rf "$EXP_DIR"
    ARGS+=(--pytorch_weight_path="$PRETRAIN_CKPT")
fi

# On the a40 nodes GPU pairs sit on different NUMA nodes (topology SYS) and NCCL's
# P2P/CUMEM transport hangs on the first collective; shared memory works
# (checked with slurm/nccl_test.sbatch).
export NCCL_P2P_DISABLE=1
# Several jobs can share a node, so let c10d pick a free rendezvous port.
ftp1_run torchrun --nproc_per_node="$NUM_GPUS" --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    scripts/zarr_train_ftp1_pytorch.py "${ARGS[@]}"

echo "=== done $(date); checkpoints: $(ls "$EXP_DIR" | tr '\n' ' ') ==="
