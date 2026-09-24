#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-models/formula}"
ENV_DIR="${ROOT}/unimumer-env"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
VLLM_VERSION="${VLLM_VERSION:-0.27.1}"
CPU_ARCH="$(uname -m)"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl"

# This machine has heterogeneous GPUs; keep CUDA index ordering stable so CUDA_VISIBLE_DEVICES
# selects the same physical adapter across PyTorch/vLLM invocations.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# Always recreate the worker environment. Reusing an environment that previously pulled cu130
# leaves incompatible torch/NVIDIA packages behind even if a later pip install requests cu129.
rm -rf "${ENV_DIR}"
python -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip

# vLLM wheels bundle tightly-coupled PyTorch/CUDA binaries. The server driver on our target
# machines exposes CUDA 12.9, so install the official cu129 stack explicitly instead of letting
# PyPI choose the newer cu130 wheel.
"${ENV_DIR}/bin/pip" install \
  "${VLLM_WHEEL}" \
  --extra-index-url "${PYTORCH_INDEX_URL}"

"${ENV_DIR}/bin/pip" install \
  "transformers>=5.0" \
  "qwen-vl-utils" \
  "Pillow>=10.0"

"${ENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import vllm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.version.cuda and not torch.version.cuda.startswith("12.9"):
    raise SystemExit(
        f"Expected a CUDA 12.9 PyTorch build for this worker, got torch.version.cuda={torch.version.cuda}"
    )
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Uni-MuMER worker environment created at: ${ENV_DIR}"
echo "Python: ${ENV_DIR}/bin/python"
