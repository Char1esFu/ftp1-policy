# Shared environment for the FTP-1 UniVTAC jobs on alex. Sourced by every sbatch
# file in this directory (and usable from a login shell: `source slurm/common.sh`).
#
# Layout
#   code       : ~/Projects/ftp1-policy                      (this repo)
#   uv env     : $WS/envs/ftp1 on the anvme workspace        (home is near its file quota)
#   raw data   : $WORK/UniVTAC/data/<task>/clean/*.hdf5     (shared with ACT / Dream-Tac, read-only here)
#   everything FTP-1 writes: $FTP1_CACHE = $WORK/ftp1_cache
#     data/UniVTAC/<name>/      parsed zarr, one dir per task
#     assets/                   norm stats + T3 encoder cache
#     checkpoints/ftp1/<exp>/   training checkpoints
#     pretrain_ckpt/            ftp1_pretrain_v0426_50kstep
#     openpi/                   OPENPI_DATA_HOME (PaliGemma tokenizer)

if [ -z "${WORK:-}" ]; then
    echo "WORK is not set; submit from a login shell" >&2
    return 1 2>/dev/null || exit 1
fi

WS_NAME="${WS_NAME:-wam-envs}"
WS_ROOT="$(ws_find "$WS_NAME" 2>/dev/null || true)"
if [ -z "$WS_ROOT" ] || [ ! -d "$WS_ROOT" ]; then
    echo "workspace '$WS_NAME' missing or expired; run: ws_allocate $WS_NAME 90" >&2
    return 3 2>/dev/null || exit 3
fi
WS="$WS_ROOT/ftp1"

export UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$WS/envs/ftp1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WS/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WS/python}"
export UV_LINK_MODE=hardlink

# Compute nodes reach the internet only through the NHR proxy.
export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"
export GIT_LFS_SKIP_SMUDGE=1

export FTP1_CACHE="${FTP1_CACHE:-$WORK/ftp1_cache}"
export UNIVTAC_RAW="${UNIVTAC_RAW:-$WORK/UniVTAC/data}"
export UNIVTAC_ZARR="${UNIVTAC_ZARR:-$FTP1_CACHE/data/UniVTAC}"
export ASSETS_BASE_DIR="$FTP1_CACHE/assets"
export CHECKPOINT_BASE_DIR="$FTP1_CACHE/checkpoints"
export PRETRAIN_CKPT="${PRETRAIN_CKPT:-$FTP1_CACHE/pretrain_ckpt/ftp1_pretrain_v0426_50kstep}"
export OPENPI_DATA_HOME="$FTP1_CACHE/openpi"
export HF_HOME="${HF_HOME:-$WORK/hf}"

# JAX is only imported for openpi data structures; keep it off the GPUs.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# torch.compile caches (thousands of small files) default to ~/.triton and
# /tmp; home is at its file quota, so keep them on the node-local $TMPDIR.
export TORCH_COMPILE_DIR="${TMPDIR:-/tmp}/torch_compile_${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="$TORCH_COMPILE_DIR/inductor"
export TRITON_CACHE_DIR="$TORCH_COMPILE_DIR/triton"
export HYDRA_FULL_ERROR=1

# Logging: openpi defaults to SwanLab (USE_SWANLAB=true); use wandb like Dream-Tac.
# Offline by default so a network hiccup cannot kill a queued job; upload with
# `bash slurm/wandb_sync.sh` (or export WANDB_MODE=online before sbatch).
export USE_SWANLAB=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$FTP1_CACHE/wandb}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-ftp1_univtac}"
mkdir -p "$WANDB_DIR"

# Released task names. put_bottle_in_shelf was trained as domain "UniVTAC_put_bottle".
FTP1_UNIVTAC_TASKS=(insert_hole insert_tube lift_bottle lift_can pull_out_key put_bottle_in_shelf)
ftp1_short_name() {
    if [ "$1" = "put_bottle_in_shelf" ]; then echo put_bottle; else echo "$1"; fi
}

# Run a command inside the project env without re-syncing it (the sync would
# undo nothing but costs minutes and needs the network).
ftp1_run() {
    "$UV_BIN" run --frozen --no-sync "$@"
}
