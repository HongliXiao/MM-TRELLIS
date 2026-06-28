#!/usr/bin/env bash
set -euo pipefail

export ATTN_BACKEND=${ATTN_BACKEND:-xformers}
export SPCONV_ALGO=${SPCONV_ALGO:-native}

python inference.py \
  --input_root examples_input/initial_test_instances \
  --pts_root examples_input/drivestudio_way_processed \
  --instances static_006_002 \
  --split test \
  --pretrained_model microsoft/TRELLIS-image-large \
  --output_dir outputs/mm_trellis_example
