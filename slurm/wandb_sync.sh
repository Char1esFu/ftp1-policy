#!/bin/bash
# Upload offline wandb runs from FTP-1 training (login node; needs the proxy).
#
#   cd ~/Projects/ftp1-policy && bash slurm/wandb_sync.sh          # all runs
#   bash slurm/wandb_sync.sh --include-synced                      # re-push finished runs
#
# Safe to re-run while a job is still training: it uploads what is there so far.

set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/common.sh

shopt -s nullglob
runs=("$WANDB_DIR"/wandb/offline-run-*)
if [ "${#runs[@]}" -eq 0 ]; then
    echo "no offline runs under $WANDB_DIR/wandb"
    exit 0
fi
ftp1_run wandb sync "$@" "${runs[@]}"
