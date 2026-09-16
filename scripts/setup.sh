#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Select ROCm wheels appropriate to YOUR GPU/driver using the official selector.
# Explicit index avoids accidentally installing CUDA wheels on an AMD machine.
: "${TORCH_INDEX_URL:?Set TORCH_INDEX_URL to the compatible PyTorch ROCm or CPU wheel index}"
if [[ -e .venv ]]; then
  echo ".venv already exists; use a fresh directory" >&2
  exit 1
fi
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url "$TORCH_INDEX_URL"
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -c 'import torch; print("torch",torch.__version__,"HIP",torch.version.hip,"GPU",torch.cuda.is_available())'
