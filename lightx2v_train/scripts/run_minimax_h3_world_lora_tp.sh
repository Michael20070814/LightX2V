#!/usr/bin/env bash
set -euo pipefail

: "${H3_MODEL_ROOT:?Set H3_MODEL_ROOT to the converted model root}"
: "${H3_CACHE_MANIFEST:?Set H3_CACHE_MANIFEST to the world cache_data.jsonl}"
: "${H3_TRAIN_OUTPUT:?Set H3_TRAIN_OUTPUT to the training output directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    train.py --config configs/train/flow/minimax_h3_world_lora_tp.yaml
