#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${PROJECT_DIR}/.venv-training"
BOOTSTRAP_DIR="${PROJECT_DIR}/.bootstrap-virtualenv"

PYTORCH_VERSION="2.4.1"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
CAUSAL_CONV1D_WHEEL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
MAMBA_SSM_WHEEL="https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

if [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
    echo "ERROR: this locked environment requires Python 3.10." >&2
    exit 1
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    # Ubuntu on this server has pip but not python3.10-venv/ensurepip. Bootstrap
    # virtualenv locally so no system packages or administrator rights are needed.
    rm -rf "${BOOTSTRAP_DIR}"
    python3 -m pip install --disable-pip-version-check --no-warn-script-location \
        --target "${BOOTSTRAP_DIR}" "virtualenv==20.27.1"
    PYTHONPATH="${BOOTSTRAP_DIR}" python3 -m virtualenv "${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade \
    "pip==24.2" "setuptools==75.1.0" "wheel==0.44.0"

# Install the CUDA-enabled PyTorch wheel first. This avoids the CPU-only wheel
# and lets the extension wheels resolve against the intended torch ABI.
"${PYTHON}" -m pip install --index-url "${PYTORCH_INDEX}" \
    "torch==${PYTORCH_VERSION}"
"${PYTHON}" -m pip install -r "${PROJECT_DIR}/requirements-training-lock.txt"

# These official prebuilt wheels are compiled for the PyTorch 2.4 ABI and CUDA
# 12.x. They avoid compiling against the server's older system CUDA toolkit.
"${PYTHON}" -m pip install --no-deps "${CAUSAL_CONV1D_WHEEL}"
"${PYTHON}" -m pip install --no-deps "${MAMBA_SSM_WHEEL}"

# Make the modules in src importable without relying on PYTHONPATH.
"${PYTHON}" -m pip install --no-deps --editable "${PROJECT_DIR}"

echo
echo "Environment ready: ${ENV_DIR}"
echo "Activate it with: source ${ENV_DIR}/bin/activate"
echo "Validate it with: ${PROJECT_DIR}/env/verify_training_env.py"
