#!/bin/bash
# One-time environment setup on an alex login node (needs the network; nothing
# compiles, so a login node is fine). Re-run after dependency changes.
#
#   cd ~/Projects/ftp1-policy && bash slurm/setup_env.sh
#
# Steps: uv sync into the anvme workspace, editable install, transformers_replace
# patch, then download the pretrained FTP-1 checkpoint, the T3 GelSight Mini
# encoder and the PaliGemma tokenizer into $FTP1_CACHE.

set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/common.sh

# The repo's uv.lock pins every wheel to the Tsinghua mirror, which times out
# from here on the large nvidia/torch wheels. Same paths and hashes exist on PyPI.
sed -i -e 's#https://pypi.tuna.tsinghua.edu.cn/simple#https://pypi.org/simple#g' \
       -e 's#https://pypi.tuna.tsinghua.edu.cn/packages/#https://files.pythonhosted.org/packages/#g' uv.lock
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

# uv sync also installs openpi itself (editable); `uv pip` ignores
# UV_PROJECT_ENVIRONMENT, hence the explicit --python.
"$UV_BIN" sync --python 3.11
"$UV_BIN" pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -e .

# openpi's PyTorch FTP-1 needs patched transformers files (FTP1Pytorch checks for them).
SITE="$("$UV_BIN" run --frozen --no-sync python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cp -r src/openpi/models_pytorch/transformers_replace/* "$SITE/transformers/"

mkdir -p "$PRETRAIN_CKPT" "$ASSETS_BASE_DIR/checkpoints_t3_encoder_base" "$OPENPI_DATA_HOME/big_vision"
if [ ! -s "$PRETRAIN_CKPT/model.safetensors" ]; then
    "$UV_BIN" run --frozen --no-sync hf download MJJJJ1064/ftp1_v0426_50kstep --local-dir "$PRETRAIN_CKPT"
fi
# T3 bootstrap weights: overwritten by the pretrained checkpoint, but the model
# constructor loads them first, and 8 ranks downloading at once would race.
# (trunk.pth 404s upstream; the constructor skips it.)
# The model's ViT is 768 wide x 3 blocks, i.e. T3-Medium, which is also what the
# pretrained GelSightMini tokenizer holds. The code's default URL points at
# t3_large (1024 wide), whose mini.pth fails to load into that shape.
T3_DIR="$ASSETS_BASE_DIR/checkpoints_t3_encoder_base"
[ -s "$T3_DIR/mini.pth" ] || curl -fL -o "$T3_DIR/mini.pth" \
    https://huggingface.co/datasets/alanz-mit/FoundationTactile/resolve/main/models/t3_medium/encoders/mini.pth
# openpi fetches the tokenizer from GCS through gcsfs, which fails here; plain HTTPS works.
TOK="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
[ -s "$TOK" ] || curl -fL -o "$TOK" https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

ftp1_run python - <<'EOF'
import torch, transformers
from transformers.models.siglip import check
from openpi.models_pytorch.ftp1_pytorch import FTP1Pytorch  # noqa: F401
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("transformers", transformers.__version__, "| replace patch:", check.check_whether_transformers_replace_is_installed_correctly())
EOF
echo "setup done"
