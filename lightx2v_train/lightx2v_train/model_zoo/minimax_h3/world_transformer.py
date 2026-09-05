"""World conditioning on the trainable Diffusers H3 modules, with optional TP."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from lightx2v_train.runtime.tensor_parallel import TensorParallelLinear, tp_rank, tp_size

_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


def fuse_qkv_weights(parts, head_dim):
    """Pack independent Q/K/V weights in H3-World's [head, QKV, dim] order."""
    return torch.stack([part.unflatten(0, (-1, head_dim)) for part in parts], dim=1).flatten(0, 2)


def action_visibility(action_rows, video_start, frame_rows, seq_len, real_used, device):
    """Directed reference visibility, including self-only padding rows."""
    annotation = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    frame = torch.full_like(annotation, -1)
    limit = torch.tensor(real_used, dtype=torch.int32, device=device)
    for index, (lo, hi) in enumerate(action_rows):
        annotation[int(lo) : int(hi)] = index
    end = video_start + len(action_rows) * frame_rows
    frame[video_start:end] = torch.arange(len(action_rows), device=device).repeat_interleave(frame_rows)

    def visible(batch, head, query, key):
        # Block-mask construction also evaluates rounded-up block coordinates.
        q, k = query.clamp(max=seq_len - 1), key.clamp(max=seq_len - 1)
        aq, ak, fq, fk = annotation[q], annotation[k], frame[q], frame[k]
        leak_out = (ak >= 0) & ~(((aq >= 0) & (aq == ak)) | ((fq >= 0) & (fq == ak)))
        leak_in = (aq >= 0) & (fk >= 0) & (aq != fk)
        real = (query < limit) & (key < limit)
        return (query == key) | (real & ~leak_out & ~leak_in)

    return visible


def build_action_mask(packed, device, backend):
    length = int(packed["seq_len"])
    visible = action_visibility(
        packed["action_text_spans_local"],
        int(packed["action_video_start"]),
        int(packed["action_frame_rows"]),
        length,
        int(packed["action_real_used"]),
        device,
    )
    if backend == "flex":
        return create_block_mask(visible, B=None, H=None, Q_LEN=length, KV_LEN=length, device=str(device), _compile=True)
    if length > 4096:
        raise ValueError("Dense SDPA is for small validation cases; use attention_backend=flex for training clips.")
    rows = torch.arange(length, device=device)
    return visible(None, None, rows[:, None], rows[None, :])


class WorldAttentionProcessor:
    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        if hasattr(attn, "qkv_proj"):
            q, k, v = attn.qkv_proj(hidden_states).unflatten(-1, (attn.heads, 3, attn.head_dim)).unbind(-2)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
            q, k, v = (x.unflatten(-1, (attn.heads, attn.head_dim)) for x in (q, k, v))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q, k = _apply_rotary_emb(q, *rotary_emb), _apply_rotary_emb(k, *rotary_emb)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        if attention_mask is not None and not torch.is_tensor(attention_mask):
            output = _compiled_flex_attention(q, k, v, block_mask=attention_mask)
        else:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        output = output.transpose(1, 2).flatten(2).to(q.dtype)
        return attn.o_proj(output) if hasattr(attn, "o_proj") else attn.to_out[1](attn.to_out[0](output))


class WorldTransformer(nn.Module):
    def __init__(self, transformer, attention_backend="flex"):
        super().__init__()
        self.base = transformer
        self.attention_backend = attention_backend
        self.gradient_checkpointing = False
        for block in transformer.transformer_blocks:
            attn = block.attn
            linear = nn.Linear(attn.to_q.in_features, 3 * attn.to_q.out_features, bias=False, device=attn.to_q.weight.device, dtype=attn.to_q.weight.dtype)
            linear.weight = nn.Parameter(fuse_qkv_weights([attn.to_q.weight, attn.to_k.weight, attn.to_v.weight], attn.head_dim))
            attn.qkv_proj, attn.o_proj = linear, attn.to_out[0]
            del attn.to_q, attn.to_k, attn.to_v, attn.to_out
            attn.set_processor(WorldAttentionProcessor())
        for block in transformer.token_refiner.refiner_blocks:
            block.attn.set_processor(WorldAttentionProcessor())

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def shard(self):
        if tp_size() == 1:
            return
        for block in [*self.base.transformer_blocks, *self.base.token_refiner.refiner_blocks]:
            attn = block.attn
            if attn.heads % tp_size():
                raise ValueError("H3 attention heads must be divisible by TP size.")
            if hasattr(attn, "qkv_proj"):
                attn.qkv_proj = TensorParallelLinear(attn.qkv_proj, column=True)
                attn.o_proj = TensorParallelLinear(attn.o_proj, column=False)
            else:
                for name in ("to_q", "to_k", "to_v"):
                    setattr(attn, name, TensorParallelLinear(getattr(attn, name), column=True))
                attn.to_out[0] = TensorParallelLinear(attn.to_out[0], column=False)
            attn.heads //= tp_size()
            attn.inner_dim //= tp_size()
            block.ff.net[0].proj = TensorParallelLinear(block.ff.net[0].proj, column=True, segments=2)
            block.ff.net[2] = TensorParallelLinear(block.ff.net[2], column=False)
            if hasattr(block, "adaln_proj"):
                block.adaln_proj.linear = TensorParallelLinear(block.adaln_proj.linear, column=True, gather_output=True)

    def forward(self, video, audio, condition, video_sigma, audio_sigma):
        base, packed = self.base, condition["packed"]
        device = video.device
        img_pos, audio_pos, text_pos = (packed[key].to(device).long() for key in ("img_pos", "audio_pos", "text_pos"))
        anchor = condition["keyframe_cond_anchor"].to(device=device, dtype=video.dtype).unsqueeze(0)
        video = torch.cat([anchor, video], dim=1)
        text = condition["prompt_embeds"].to(device=device, dtype=base.context_embedder.weight.dtype).unsqueeze(0)
        text = base.context_embedder(text)
        bounds = packed["refiner_cu_seqlens"].tolist()
        text = torch.cat([base.token_refiner(text[:, lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:])], dim=1)
        hidden = text.new_zeros((1, int(packed["seq_len"]), text.shape[-1]))
        hidden = hidden.index_copy(1, text_pos, text)
        hidden = hidden.index_copy(1, img_pos, base.proj_in(video.to(base.proj_in.weight.dtype)).to(text.dtype))
        hidden = hidden.index_copy(1, audio_pos, base.audio_proj_in(audio.to(base.audio_proj_in.weight.dtype)).to(text.dtype))
        times = torch.ones(hidden.shape[1], device=device, dtype=torch.float32) * (1 - video_sigma)
        times[audio_pos] = 1 - audio_sigma
        times[img_pos[: anchor.shape[1]]] = torch.maximum(1 - video_sigma, times.new_tensor(condition["imgvid_cond_noise_aug"]))
        unique, inverse = torch.unique(times, sorted=True, return_inverse=True)
        temb = base.time_embedder(base.time_proj(unique).to(base.time_embedder.linear_1.weight.dtype))
        indices = inverse * 3 + packed["token_tags"].to(device).clamp(min=0)
        positions = packed["img_position_ids"].to(device).reshape(-1, 3)
        rotary = base.rope(positions)
        mask = condition.get("attention_mask")
        if mask is None:
            mask = build_action_mask(packed, device, self.attention_backend)
        for block in base.transformer_blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, temb, indices, rotary, mask, use_reentrant=False)
            else:
                hidden = block(hidden, temb, indices, rotary, mask)
        # Only target rows need output projections; anchors and padding carry no loss.
        video_pos = img_pos[anchor.shape[1] :]
        video_hidden = base.norm_out(hidden.index_select(1, video_pos), temb, inverse[video_pos])
        audio_hidden = base.norm_out(hidden.index_select(1, audio_pos), temb, inverse[audio_pos])
        return base.proj_out(video_hidden.to(base.proj_out.weight.dtype)), base.audio_proj_out(audio_hidden.to(base.audio_proj_out.weight.dtype))


def load_world_weights(transformer, directory, device, dtype):
    """Stream only local shards from safetensors into a meta-initialized model."""
    directory = Path(directory)
    index = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        filename = "diffusion_pytorch_model.safetensors"
        with safe_open(directory / filename, framework="pt", device="cpu") as handle:
            weight_map = {key: filename for key in handle.keys()}

    def read(name, spec=None):
        if name not in weight_map:
            raise KeyError(f"Missing pretrained H3 tensor: {name}")
        with safe_open(directory / weight_map[name], framework="pt", device="cpu") as handle:
            source = handle.get_slice(name)
            if spec is None:
                return source[:]
            dim, segments = spec
            shape = source.get_shape()
            segment = shape[dim] // segments
            if shape[dim] % (segments * tp_size()):
                raise ValueError(f"Cannot TP-shard pretrained tensor {name}: {shape}.")
            width = segment // tp_size()
            pieces = []
            for s in range(segments):
                selection = [slice(None)] * len(shape)
                start = s * segment + tp_rank() * width
                selection[dim] = slice(start, start + width)
                pieces.append(source[tuple(selection)])
            return torch.cat(pieces, dim=dim)

    for name, param in list(transformer.named_parameters()):
        spec = getattr(param, "_tp_shard", None)
        if "lora_" in name:
            value = torch.empty(param.shape, device=device, dtype=torch.float32)
            if "lora_A" in name:
                # Initialize the full small adapter before slicing, identical on all ranks.
                full_shape = list(param.shape)
                if spec:
                    full_shape[spec[0]] *= tp_size()
                from lightx2v_train.runtime.tensor_parallel import shard_tensor

                full = torch.empty(full_shape, dtype=torch.float32).normal_(std=1 / full_shape[0])
                value.copy_(shard_tensor(full, *spec) if spec else full)
            else:
                value.zero_()
        else:
            key = name.removeprefix("base.").replace(".module.", ".").replace(".base_layer.", ".")
            key = key.replace(".o_proj.", ".to_out.0.")
            if ".qkv_proj." in key:
                parts = [read(key.replace(".qkv_proj.", f".to_{letter}."), (0, 1) if spec else None) for letter in "qkv"]
                value = fuse_qkv_weights(parts, transformer.base.config.attention_head_dim)
            else:
                value = read(key, spec)
            target_dtype = torch.float32 if any(part in key for part in transformer.base._keep_in_fp32_modules) else dtype
            value = value.to(device=device, dtype=target_dtype)
        if value.shape != param.shape:
            raise ValueError(f"Pretrained shape mismatch for {name}: {value.shape} != {param.shape}.")
        replacement = nn.Parameter(value, requires_grad=param.requires_grad)
        replacement.__dict__.update(param.__dict__)
        parent, attr = name.rsplit(".", 1)
        setattr(transformer.get_submodule(parent), attr, replacement)
    # RoPE's derived buffer is not part of the safetensors state dict.
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    config = transformer.base.config
    transformer.base.rope = MiniMaxH3RotaryPosEmbed(config.rope_freq_dim, config.rope_theta).to(device)
