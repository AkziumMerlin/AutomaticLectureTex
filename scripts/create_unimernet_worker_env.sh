#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-models/formula}"
ENV_DIR="${ROOT}/unimernet-env"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

python -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip

# Keep UniMERNet's old Transformers dependency isolated from qwen-asr in the main lecture env.
"${ENV_DIR}/bin/pip" install   --index-url "${PYTORCH_INDEX_URL}"   "torch==2.10.0"   "torchvision==0.25.0"

"${ENV_DIR}/bin/pip" install "unimernet==0.2.3"

echo "UniMERNet worker environment created at: ${ENV_DIR}"
echo "Python: ${ENV_DIR}/bin/python"
