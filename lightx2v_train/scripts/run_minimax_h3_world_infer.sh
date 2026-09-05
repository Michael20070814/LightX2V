#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${H3_MODEL_ROOT:?Set H3_MODEL_ROOT to the converted Diffusers model root}"
: "${H3_LORA_PATH:?Set H3_LORA_PATH to a world LoRA directory or safetensors file}"
: "${H3_FIRST_FRAME:?Set H3_FIRST_FRAME to an image path}"
: "${H3_SCENE_PROMPT:?Set H3_SCENE_PROMPT to the scene description}"
: "${H3_INFER_OUTPUT:?Set H3_INFER_OUTPUT to the output directory}"

exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    infer.py --config "${H3_INFER_CONFIG:-configs/infer/minimax_h3_world_lora.yaml}"
