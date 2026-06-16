#!/usr/bin/env bash
#
# Install dependencies for gpt-oss RMSNorm fusion on a TPU v6e VM.
#
# Usage (after `ssh`-ing into the TPU VM):
#     git clone <this-repo> && cd gptoss-tpu-fusion
#     bash scripts/install_deps.sh
#     huggingface-cli login          # needed to pull openai/gpt-oss-20b
#     python scripts/fuse_gpt_oss.py --model-id openai/gpt-oss-20b \
#         --output-dir gpt-oss-20b-fused-bf16
#
# Notes
#   * Pins are "known-good recent" — bump if your TPU runtime needs a different
#     torch_xla. torch and torch_xla versions MUST match (same major.minor).
#   * gpt-oss needs a recent transformers (>= 4.55, when gpt-oss landed).
#   * MXFP4 has no TPU compute path; we load gpt-oss dequantized to BF16, so no
#     Triton/MXFP4 GPU kernels are installed here.

set -euo pipefail

PY="${PYTHON:-python3}"
echo "==> Using interpreter: $($PY --version 2>&1)"

# Optional but recommended: isolated venv
if [ "${USE_VENV:-1}" = "1" ]; then
  echo "==> Creating virtualenv .venv"
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
fi

echo "==> Upgrading pip"
"$PY" -m pip install --upgrade pip wheel

echo "==> Installing PyTorch/XLA for TPU"
# torch + torch_xla[tpu] from the libtpu wheel indexes.
"$PY" -m pip install 'torch~=2.8.0' 'torch_xla[tpu]~=2.8.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html

echo "==> Installing HuggingFace + model deps"
"$PY" -m pip install \
  'transformers>=4.55' \
  'accelerate>=0.34' \
  'safetensors>=0.4' \
  'huggingface_hub[cli]>=0.24' \
  sentencepiece

echo "==> Sanity check: torch_xla can see the TPU"
PJRT_DEVICE=TPU "$PY" - <<'PYEOF'
import torch_xla.core.xla_model as xm
dev = xm.xla_device()
print("XLA device:", dev)
print("World size:", xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else "n/a")
PYEOF

echo "==> Done. Next:"
echo "    huggingface-cli login"
echo "    python scripts/fuse_gpt_oss.py --model-id openai/gpt-oss-20b --output-dir gpt-oss-20b-fused-bf16"
