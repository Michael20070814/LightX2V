"""Strict import of LightX2V and H3-World text-action LoRA adapters."""

import json
import math
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


def read_world_lora(path, transformer, *, alpha=None):
    path = Path(path).expanduser()
    weights = path / "pytorch_lora_weights.safetensors" if path.is_dir() else path
    state = load_file(str(weights), device="cpu")
    normalized, formats = {}, set()
    for key, value in state.items():
        name = key
        for prefix in ("base_model.model.", "pipe.dit.", "dit."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        native = re.fullmatch(r"base\.transformer_blocks\.(\d+)\.attn\.(qkv_proj|o_proj)\.lora_([AB])(?:\.default)?\.weight", name)
        reference = re.fullmatch(r"blocks\.(\d+)\.attn\.(qkv_proj|out_proj)\.lora_([AB])(?:\.default)?\.weight", name)
        match = native or reference
        if match is None:
            raise ValueError(f"Unsupported world LoRA tensor: {key}. Only backbone text-action attention LoRA is supported.")
        formats.add("native" if native else "h3_world")
        block, projection, factor = match.groups()
        projection = "o_proj" if projection == "out_proj" else projection
        target = f"base.transformer_blocks.{block}.attn.{projection}.lora_{factor}.default.weight"
        if target in normalized:
            raise ValueError(f"Duplicate world LoRA tensor: {target}.")
        if value.ndim != 2 or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"World LoRA tensor must be a finite floating-point matrix: {key}.")
        normalized[target] = value
    if len(formats) != 1:
        raise ValueError("World LoRA must contain one nonempty, consistent adapter format.")
    expected, ranks = set(), set()
    for name, module in transformer.named_modules():
        if re.fullmatch(r"base\.transformer_blocks\.\d+\.attn\.(qkv_proj|o_proj)", name):
            a_key, b_key = (f"{name}.lora_{factor}.default.weight" for factor in "AB")
            expected.update((a_key, b_key))
            if a_key not in normalized or b_key not in normalized:
                raise ValueError(f"Missing world LoRA A/B pair for {name}.")
            a, b = normalized[a_key], normalized[b_key]
            rank = a.shape[0]
            if rank <= 0 or a.shape[1] != module.in_features or b.shape != (module.out_features, rank):
                raise ValueError(f"World LoRA shape mismatch for {name}.")
            ranks.add(rank)
    if expected != normalized.keys() or len(ranks) != 1:
        raise ValueError("World LoRA has unexpected layers or inconsistent ranks.")
    rank = ranks.pop()
    if formats == {"native"}:
        metadata_path = weights.parent / "adapter_config.json"
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format") != "lightx2v_h3_world_v1" or metadata.get("qkv_layout") != "head_qkv_dim":
            raise ValueError("World LoRA QKV layout must be head_qkv_dim; old checkpoints need a separate conversion.")
        if metadata.get("rank") != rank or set(metadata.get("target_modules", [])) != {"qkv_proj", "o_proj"}:
            raise ValueError("World LoRA metadata does not match its tensors.")
        stored_alpha = float(metadata["alpha"])
        if alpha is not None and float(alpha) != stored_alpha:
            raise ValueError("Configured alpha differs from the world checkpoint; use strength to scale the adapter.")
        alpha = stored_alpha
    elif alpha is None:
        # The released H3-World inference script applies B @ A with unit scale.
        alpha = rank
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("World LoRA alpha must be finite and positive.")
    return normalized, rank, alpha
