"""Autograd collectives and linear sharding for replicated-sequence TP training."""

import torch
import torch.distributed as dist
from torch import nn

from .distributed import get_tensor_parallel_group, get_tensor_parallel_world_size


def tp_size():
    return get_tensor_parallel_world_size()


def tp_rank():
    return dist.get_rank(get_tensor_parallel_group()) if tp_size() > 1 else 0


def _sum(value):
    if tp_size() > 1:
        dist.all_reduce(value, group=get_tensor_parallel_group())
    return value


class _Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return _sum(grad.contiguous().clone())


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _sum(x.contiguous().clone())

    @staticmethod
    def backward(ctx, grad):
        return grad


def shard_tensor(value, dim, segments=1):
    parts = value.chunk(segments, dim=dim)
    if any(part.shape[dim] % tp_size() for part in parts):
        raise ValueError("Linear dimensions must be divisible by tensor parallel size.")
    return torch.cat([part.chunk(tp_size(), dim=dim)[tp_rank()] for part in parts], dim=dim).contiguous()


def gather_tensor(value, dim, segments=1):
    if tp_size() == 1:
        return value
    pieces = [torch.empty_like(value) for _ in range(tp_size())]
    dist.all_gather(pieces, value.contiguous(), group=get_tensor_parallel_group())
    chunks = [piece.chunk(segments, dim=dim) for piece in pieces]
    return torch.cat([torch.cat([chunk[s] for chunk in chunks], dim=dim) for s in range(segments)], dim=dim)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, segments):
        ctx.segments = segments
        return gather_tensor(x, -1, segments)

    @staticmethod
    def backward(ctx, grad):
        return shard_tensor(grad, -1, ctx.segments), None


class TensorParallelLinear(nn.Module):
    """Shard an ordinary or PEFT linear, retaining fused projection boundaries."""

    def __init__(self, linear, *, column, segments=1, gather_output=False):
        super().__init__()
        self.module = linear
        self.column = column
        self.segments = segments
        self.gather_output = gather_output
        base = linear.get_base_layer() if hasattr(linear, "get_base_layer") else linear
        self._shard(base, "weight", 0 if column else 1, segments if column else 1)
        if base.bias is not None:
            if column:
                self._shard(base, "bias", 0, segments)
            else:
                raise ValueError("Row parallel linears must be bias-free.")
        base.out_features = base.weight.shape[0]
        base.in_features = base.weight.shape[1]
        if hasattr(linear, "lora_A"):
            for adapter in linear.lora_A:
                a, b = linear.lora_A[adapter], linear.lora_B[adapter]
                self._shard(b if column else a, "weight", 0 if column else 1, segments if column else 1)
                replicated = a.weight if column else b.weight
                replicated._tp_replicated_partial = True

    @staticmethod
    def _shard(module, name, dim, segments):
        old = getattr(module, name)
        param = nn.Parameter(shard_tensor(old.detach(), dim, segments), requires_grad=old.requires_grad)
        param._tp_shard = (dim, segments)
        setattr(module, name, param)

    def forward(self, x):
        if self.column:
            result = self.module(_Copy.apply(x))
            return _Gather.apply(result, self.segments) if self.gather_output else result
        return _Reduce.apply(self.module(x))


def sync_tensor_parallel_gradients(params):
    for param in params:
        if param.grad is not None and getattr(param, "_tp_replicated_partial", False):
            _sum(param.grad)


def clip_tensor_parallel_grad_norm_(params, max_norm):
    params = list(params)
    if tp_size() == 1:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    norm_squared = torch.zeros((), device=params[0].device, dtype=torch.float32)
    for param in params:
        if param.grad is not None:
            divisor = 1 if hasattr(param, "_tp_shard") else tp_size()
            norm_squared += param.grad.float().square().sum() / divisor
    total_norm = _sum(norm_squared).sqrt()
    if not torch.isfinite(total_norm):
        raise RuntimeError("Non-finite tensor parallel gradient norm.")
    scale = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for param in params:
        if param.grad is not None:
            param.grad.mul_(scale)
    return total_norm


def broadcast_tensor_parallel_value(value):
    if tp_size() == 1:
        return value
    if torch.is_tensor(value):
        dist.broadcast(value, src=0, group=get_tensor_parallel_group())
        return value
    if isinstance(value, dict):
        return {key: broadcast_tensor_parallel_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(broadcast_tensor_parallel_value(item) for item in value)
    objects = [value if tp_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0, group=get_tensor_parallel_group())
    return objects[0]
