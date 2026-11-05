#!/usr/bin/env bash
# Builds the `uni` conda env exactly as UniVTAC/Installation_FTP1.md describes,
# without sudo (gcc 11.4 / cmake 3.26 / CUDA 12.4 / ffmpeg come from conda).
# Stages stamp $DEPS_DIR/.state and are skipped on re-run.
#
#   bash UniVTAC/scripts/shell/setup_uni_env.sh            # from the ftp1-policy root
#   UNI_FORCE=tacex_uipc bash UniVTAC/scripts/shell/...     # redo one stage (comma list, or "all")
#
# IsaacLab and cuRobo are cloned into $DEPS_DIR rather than reusing other
# checkouts: editable installs compile into the source tree against this env's
# torch (2.7.0+cu128, see pin_torch).

set -Eeo pipefail

FTP1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UNIVTAC_ROOT="$FTP1_ROOT/UniVTAC"
TACEX_ROOT="$UNIVTAC_ROOT/third_party/TacEx"
ENV_NAME="${ENV_NAME:-uni}"
DEPS_DIR="${DEPS_DIR:-$HOME/proj/ftp1-uni-deps}"
VCPKG_DIR="${VCPKG_DIR:-$HOME/Toolchain/vcpkg}"
CUROBO_TAG="${CUROBO_TAG:-v0.7.8}"
STATE_DIR="$DEPS_DIR/.state"
CONDA_BASE="$(conda info --base)"

# An activated uv venv (e.g. this repo's .venv) wins over conda in CMake's
# FindPython, and libuipc then runs its generator scripts with that interpreter.
unset VIRTUAL_ENV
PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.venv/bin$' | paste -sd:)"

mkdir -p "$DEPS_DIR" "$STATE_DIR"
exec > >(tee -a "$STATE_DIR/setup.log") 2>&1

stage() {
    local name="$1"; shift
    local force="${UNI_FORCE:-}"
    if [ -f "$STATE_DIR/$name.done" ] && [[ ",$force," != *",all,"* && ",$force," != *",$name,"* ]]; then
        echo "--- $name: done, skipping"; return 0
    fi
    printf '\n>>> [%s] %s\n' "$(date +%H:%M:%S)" "$name"
    "$@"
    touch "$STATE_DIR/$name.done"
    printf '<<< [%s] %s ok\n' "$(date +%H:%M:%S)" "$name"
}

activate() {
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
}

# Step 2
s_conda_env() {
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda env list | grep -qE "^${ENV_NAME}\s" || CONDA_SOLVER=classic conda create -n "$ENV_NAME" python=3.10 -y
}

# Step 3
s_isaacsim() {
    activate
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
    pip install --upgrade pip
    pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
    # Headless smoke test in place of the interactive `isaacsim --allow-root --headless`.
    OMNI_KIT_ACCEPT_EULA=YES timeout 900 python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True}); print('isaacsim headless launch ok'); app.close()"
}

# Step 4 (flatdict fix applied before the install so it does not fail first)
s_isaaclab() {
    activate
    [ -d "$DEPS_DIR/IsaacLab/.git" ] || git clone https://github.com/isaac-sim/IsaacLab "$DEPS_DIR/IsaacLab"
    cd "$DEPS_DIR/IsaacLab"
    git fetch origin tag v2.1.1 && git checkout v2.1.1
    sed -i 's/flatdict==4.0.1/flatdict==4.1.0/' source/isaaclab/setup.py
    ./isaaclab.sh --install
    pip install --upgrade setuptools
    pip install -e source/isaaclab
    pin_torch
}

# isaaclab.sh --install swaps the Step 3 torch for 2.7.0+cu128 (its hard-coded
# version), but isaaclab_rl's unpinned stable-baselines3>=2.6 now resolves to a
# release that drags torch to the latest. Restore what the install produced when
# v2.1.1 was current; later stages compile against this torch.
pin_torch() {
    pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    pip install 'stable-baselines3==2.7.0'
    python -c "import torch; v = torch.__version__; assert v == '2.7.0+cu128', v; print('torch', v)"
}

# Step 5 + Step 8. The doc edits ../pyproject.toml to requires-python >=3.10 for
# the pip install; restore it afterwards so the repo's uv env and lockfile stay valid.
s_ftp1_infer() {
    activate
    cd "$FTP1_ROOT"
    cp pyproject.toml "$STATE_DIR/pyproject.toml.bak"
    sed -i '6s/.*/requires-python = ">=3.10"/' pyproject.toml
    local rc=0
    (bash UniVTAC/scripts/shell/install_ftp1_infer_into_isaacsim.sh) || rc=$?
    cp "$STATE_DIR/pyproject.toml.bak" pyproject.toml
    [ "$rc" -eq 0 ] || return "$rc"
    cp -r ./src/openpi/models_pytorch/transformers_replace/* "$CONDA_PREFIX/lib/python3.10/site-packages/transformers/"
}

# Step 6.1
s_tacex_core() {
    activate
    export ISAACLAB_PATH="$DEPS_DIR/IsaacLab"
    cd "$TACEX_ROOT"
    bash ./tacex.sh -i  # committed without the exec bit
    python -c "import torch; v = torch.__version__; assert v == '2.7.0+cu128', v"
}

# Step 6.2.1-6.2.3 + Step 9 (ffmpeg from conda-forge instead of apt)
s_uipc_deps() {
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    [ -x "$VCPKG_DIR/vcpkg" ] || { git clone https://github.com/microsoft/vcpkg.git "$VCPKG_DIR"; "$VCPKG_DIR/bootstrap-vcpkg.sh" -disableMetrics; }
    "$VCPKG_DIR/vcpkg" install spdlog:x64-linux
    conda env update -n "$ENV_NAME" --file "$TACEX_ROOT/source/tacex_uipc/libuipc/conda/env.yaml"
    # Unconstrained, the solver picks a 2016 ffmpeg 2.8.6 that needs libx264.so.138.
    conda install -n "$ENV_NAME" -c conda-forge -y 'ffmpeg>=6'
}

# GitHub re-generated the v2.9.3 tarball, so its bytes no longer match the SHA512
# in libuipc's pinned vcpkg baseline. The extracted tree is unchanged (after the
# port's patch, tiny_gltf.h is identical to an Aug 2026 build), so overlay the
# baseline's port with only the hash updated.
tinygltf_overlay() {
    local baseline old new dst="$DEPS_DIR/vcpkg-overlay/tinygltf"
    baseline="$(sed -n "s/^VCPKG_BASE_LINE = '\(.*\)'/\1/p" "$TACEX_ROOT/source/tacex_uipc/libuipc/scripts/gen_vcpkg_json.py")"
    old=4f4d479a8ad8dd858340b0bfa5af4fdc9073279e59c4240918d5dfce94d2b50b87bc0acad0a2e7659d090dd4aa3b34b456550749fe57bb4f7b58ac2f2b6927aa
    new=6dbcff3ea602d0aa45ddd87a87d32ab5ab5453901891dbccfbc660746fe11c5bd814d6f74707244351dd6326e17f6d9ad7c384417db126122cc4a2cba20b205c
    mkdir -p "$dst"
    git -C "$VCPKG_DIR" show "$baseline:ports/tinygltf/vcpkg.json" > "$dst/vcpkg.json"
    git -C "$VCPKG_DIR" show "$baseline:ports/tinygltf/portfile.cmake" | sed "s/$old/$new/" > "$dst/portfile.cmake"
    grep -q "$new" "$dst/portfile.cmake"
}

# Step 6.2.4
s_tacex_uipc() {
    activate
    tinygltf_overlay
    export VCPKG_OVERLAY_PORTS="$DEPS_DIR/vcpkg-overlay"
    export CMAKE_TOOLCHAIN_FILE="$VCPKG_DIR/scripts/buildsystems/vcpkg.cmake"
    cd "$TACEX_ROOT"
    # libuipc's tests/examples/benchmarks default ON, and linking Catch2 from vcpkg
    # fails against the conda sysroot (glibc 2.28: no __libc_single_threaded).
    # They are not part of the pyuipc module, so turn them off in the cache that
    # setup.py's cmake call reuses (it never passes these options itself).
    local cache=source/tacex_uipc/build/CMakeCache.txt o
    mkdir -p source/tacex_uipc/build
    for o in EXAMPLES TESTS BENCHMARKS; do
        if grep -q "^UIPC_BUILD_$o:" "$cache" 2>/dev/null; then
            sed -i "s/^UIPC_BUILD_$o:.*/UIPC_BUILD_$o:BOOL=0/" "$cache"
        else
            echo "UIPC_BUILD_$o:BOOL=0" >> "$cache"
        fi
    done
    pip install -e source/tacex_uipc -v
}

# Step 7. cuRobo compiles CUDA extensions against torch (cu128); the env's conda
# CUDA 12.4 toolkit from libuipc's env.yaml has the same major version.
s_curobo() {
    activate
    [ -d "$DEPS_DIR/curobo/.git" ] || git clone https://github.com/NVlabs/curobo.git "$DEPS_DIR/curobo"
    cd "$DEPS_DIR/curobo"
    git fetch --tags && git checkout "$CUROBO_TAG"
    CUDA_HOME="$CONDA_PREFIX" pip install -e . --no-build-isolation
}

s_verify() {
    activate
    cd "$FTP1_ROOT"
    JAX_PLATFORMS=cpu OMNI_KIT_ACCEPT_EULA=YES python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
assert torch.__version__ == "2.7.0+cu128" and torch.cuda.is_available()
from transformers.models.siglip import check
print("transformers", transformers.__version__, "replace patch", check.check_whether_transformers_replace_is_installed_correctly())
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper  # noqa: F401
print("FTP1 inference import ok")
import isaacsim, curobo  # noqa: F401
print("isaacsim, curobo import ok")
EOF
    # isaaclab / tacex need a running SimulationApp to import their sim modules.
    OMNI_KIT_ACCEPT_EULA=YES timeout 900 python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import isaaclab, tacex, tacex_uipc, tacex_assets
print('isaaclab, tacex, tacex_uipc, tacex_assets import ok', tacex_assets.TACEX_ASSETS_DATA_DIR, flush=True)  # close() exits hard
app.close()"
    ffmpeg -version | head -1
}

echo "=== uni env setup | ftp1 root $FTP1_ROOT | deps $DEPS_DIR | $(date) ==="
stage conda_env   s_conda_env
stage isaacsim    s_isaacsim
stage isaaclab    s_isaaclab
stage ftp1_infer  s_ftp1_infer
stage tacex_core  s_tacex_core
stage uipc_deps   s_uipc_deps
stage tacex_uipc  s_tacex_uipc
stage curobo      s_curobo
stage verify      s_verify
echo "=== uni env ready: conda activate $ENV_NAME ==="
