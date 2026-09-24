#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-models/formula}"
ENV_DIR="${ROOT}/unimumer-env"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# Recreate the isolated OCR environment. It intentionally does not contain vLLM: the OCR worker
# handles one crop at a time and uses Transformers directly to minimize VRAM overhead.
rm -rf "${ENV_DIR}"
python -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip

"${ENV_DIR}/bin/pip" install \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==2.9.1" \
  "torchvision==0.24.1"

"${ENV_DIR}/bin/pip" install \
  "transformers>=5.17,<6" \
  "accelerate>=1.10,<2" \
  "bitsandbytes>=0.49,<0.50" \
  "Pillow>=10.0"

"${ENV_DIR}/bin/python" - <<'PY'
import bitsandbytes
import torch
import transformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.version.cuda and not torch.version.cuda.startswith("12.9"):
    raise SystemExit(
        f"Expected a CUDA 12.9 PyTorch build for this worker, got torch.version.cuda={torch.version.cuda}"
    )
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Uni-MuMER low-memory worker environment created at: ${ENV_DIR}"
echo "Python: ${ENV_DIR}/bin/python"
