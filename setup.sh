#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: ./setup.sh [OPTIONS]

Options:
  --basic            Install core Python dependencies from requirements.txt
  --xformers         Install xFormers attention backend used by the verified setup
  --flash-attn       Install flash-attn attention backend (optional; may not support all GPUs)
  --spconv           Install spconv package matching the PyTorch CUDA wheel
  --mipgaussian      Install diff-gaussian-rasterization from mip-splatting
  --kaolin           Try installing NVIDIA Kaolin; manual install may be needed
  --nvdiffrast       Install nvdiffrast
  --diffoctreerast   Install diffoctreerast from source
  --pytorch3d        Install PyTorch3D
  -h, --help         Show this help

Verified environment:
  NVIDIA H20, driver 535.161.07, system CUDA 12.2
  Python 3.12.3, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126

Install PyTorch first. Example:
  pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
EOF
}

if [[ $# -eq 0 ]]; then
  show_help
  exit 0
fi

CUDA_VERSION=$(python - <<'PY'
import torch
print(torch.version.cuda or '')
PY
)
CUDA_MAJOR=${CUDA_VERSION%%.*}
CUDA_MINOR=$(python - <<'PY'
import torch
v = torch.version.cuda or ''
parts = v.split('.')
print(parts[1] if len(parts) > 1 else '')
PY
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --basic)
      pip install -r requirements.txt
      ;;
    --xformers)
      pip install xformers==0.0.31.post1
      ;;
    --flash-attn)
      pip install flash-attn --no-build-isolation
      ;;
    --spconv)
      if [[ "$CUDA_MAJOR" == "11" ]]; then
        pip install spconv-cu118
      elif [[ "$CUDA_MAJOR" == "12" && "$CUDA_MINOR" == "6" ]]; then
        pip install spconv-cu126==2.3.8 cumm-cu126==0.7.11
      elif [[ "$CUDA_MAJOR" == "12" ]]; then
        pip install spconv-cu120
      else
        echo "Unsupported CUDA version for spconv: ${CUDA_VERSION}" >&2
        exit 1
      fi
      ;;
    --mipgaussian)
      tmpdir=$(mktemp -d)
      git clone https://github.com/autonomousvision/mip-splatting.git "$tmpdir/mip-splatting"
      pip install "$tmpdir/mip-splatting/submodules/diff-gaussian-rasterization/"
      ;;
    --kaolin)
      pip install kaolin ||         echo "Kaolin wheel install failed. Please install Kaolin manually for your PyTorch/CUDA version."
      ;;
    --nvdiffrast)
      pip install nvdiffrast==0.3.3 || {
        tmpdir=$(mktemp -d)
        git clone https://github.com/NVlabs/nvdiffrast.git "$tmpdir/nvdiffrast"
        pip install "$tmpdir/nvdiffrast"
      }
      ;;
    --diffoctreerast)
      tmpdir=$(mktemp -d)
      git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git "$tmpdir/diffoctreerast"
      pip install "$tmpdir/diffoctreerast"
      ;;
    --pytorch3d)
      pip install pytorch3d==0.7.9 || pip install 'git+https://github.com/facebookresearch/pytorch3d.git'
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
  shift
done
