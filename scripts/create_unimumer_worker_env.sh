#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-models/formula}"
ENV_DIR="${ROOT}/unimumer-env"

python -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip

# Uni-MuMER's current Qwen3.5 checkpoints are supported by recent vLLM releases.
# Keep this isolated from the main lecture environment so Qwen-ASR / torch pins cannot conflict.
"${ENV_DIR}/bin/pip" install \
  "vllm>=0.27,<0.28" \
  "transformers>=5.0" \
  "qwen-vl-utils" \
  "Pillow>=10.0"

"${ENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import vllm

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Uni-MuMER worker environment created at: ${ENV_DIR}"
echo "Python: ${ENV_DIR}/bin/python"
