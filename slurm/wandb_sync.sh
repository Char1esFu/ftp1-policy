#!/bin/bash
# Upload offline wandb runs from FTP-1 training (login node; needs the proxy).
#
#   cd ~/Projects/ftp1-policy && bash slurm/wandb_sync.sh          # all runs not yet synced
#   bash slurm/wandb_sync.sh <offline-run dir>                      # one run (already-synced ones need
#                                                                   #   their run-*.wandb.synced removed)
# Live tracking of a running job (keeps uploading until the run ends):
#   nohup bash slurm/wandb_sync.sh --live <offline-run dir> > ~/wandb_live_<jobid>.log 2>&1 &
#
# Uses `wandb beta sync` with an explicit entity/project: the legacy `wandb sync` printed
# "done" for these offline runs (no entity recorded) but created nothing on the server.

set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/common.sh

shopt -s nullglob
args=() runs=()
for a in "$@"; do if [ -d "$a" ]; then runs+=("$a"); else args+=("$a"); fi; done
[ "${#runs[@]}" -gt 0 ] || runs=("$WANDB_DIR"/wandb/offline-run-*)
if [ "${#runs[@]}" -eq 0 ]; then
    echo "no offline runs under $WANDB_DIR/wandb"
    exit 0
fi
ftp1_run wandb beta sync -e "$WANDB_ENTITY" -p "$WANDB_PROJECT_NAME" "${args[@]}" "${runs[@]}"
