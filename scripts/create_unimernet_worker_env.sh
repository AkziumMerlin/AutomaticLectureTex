#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-models/formula}"
ENV_DIR="${ROOT}/unimernet-env"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

python -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip

# Keep UniMERNet's old Transformers dependency isolated from qwen-asr in the main lecture env.
"${ENV_DIR}/bin/pip" install   --index-url "${PYTORCH_INDEX_URL}"   "torch==2.10.0"   "torchvision==0.25.0"

UNIMERNET_SRC="${ROOT}/UniMERNet-src"
if [[ -d "${UNIMERNET_SRC}/.git" ]]; then
  git -C "${UNIMERNET_SRC}" fetch --depth 1 origin tag 0.2.3
  git -C "${UNIMERNET_SRC}" checkout --force 0.2.3
else
  rm -rf "${UNIMERNET_SRC}"
  git clone --depth 1 --branch 0.2.3 https://github.com/opendatalab/UniMERNet.git "${UNIMERNET_SRC}"
fi

# Install runtime dependencies first, then install the actual source tree. Some PyPI/mirror
# combinations have been observed to leave only distribution metadata without an importable
# `unimernet` package.
"${ENV_DIR}/bin/pip" install --upgrade "unimernet[full]==0.2.3"
"${ENV_DIR}/bin/pip" uninstall -y unimernet
"${ENV_DIR}/bin/pip" install --no-deps --editable "${UNIMERNET_SRC}"

"${ENV_DIR}/bin/python" - <<'PY'
import importlib.metadata
import pathlib
import unimernet
import torch

print("unimernet:", importlib.metadata.version("unimernet"))
print("unimernet module:", pathlib.Path(unimernet.__file__).resolve())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

echo "UniMERNet worker environment created at: ${ENV_DIR}"
echo "Python: ${ENV_DIR}/bin/python"
