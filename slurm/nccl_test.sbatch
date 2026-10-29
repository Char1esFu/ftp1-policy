#!/bin/bash
#SBATCH --job-name=nccl_test
#SBATCH --account=g107ea
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:2
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/nccl_%j.out
#SBATCH --error=slurm/logs/nccl_%j.out
# Diagnose multi-GPU NCCL hangs: run a tiny all_gather/all_reduce with default
# settings and with P2P disabled, each under a 120 s timeout.
#
#   cd ~/Projects/ftp1-policy && sbatch slurm/nccl_test.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/common.sh

nvidia-smi topo -m
cat > /tmp/nccl_test_$SLURM_JOB_ID.py <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(1 << 20, device="cuda") * (r + 1)
dist.all_reduce(x); torch.cuda.synchronize()
out = [torch.empty(1, device="cuda") for _ in range(dist.get_world_size())]
dist.all_gather(out, torch.tensor([float(r)], device="cuda")); torch.cuda.synchronize()
print(f"rank {r} OK all_reduce={x[0].item()} all_gather={[o.item() for o in out]}", flush=True)
dist.destroy_process_group()
PY
for v in "" "NCCL_P2P_DISABLE=1" "NCCL_P2P_LEVEL=LOC NCCL_SHM_DISABLE=1"; do
    echo "===== variant: ${v:-default} ====="
    env $v NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM timeout 120 \
        "$UV_BIN" run --frozen --no-sync torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
        /tmp/nccl_test_$SLURM_JOB_ID.py 2>&1 | grep -E 'OK all_reduce|Channel|via|NCCL version|Error|error' | head -20
    echo "exit ${PIPESTATUS[0]}"
done
