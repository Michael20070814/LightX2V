import torch
import triton
import triton.language as tl


@triton.jit
def _fused_wan_qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    freqs_ptr,
    positions_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    is_k = tl.program_id(1)
    x_ptr = k_ptr if is_k else q_ptr
    weight_ptr = k_weight_ptr if is_k else q_weight_ptr

    lanes = tl.arange(0, 32)
    lane_sum = tl.zeros((32,), tl.float32)
    for component in range(8):
        component_sum = tl.zeros((32,), tl.float32)
        for chunk in range(20):
            reduce_offsets = ((chunk * 32 + lanes) * 8) + component
            reduce_x = tl.load(x_ptr + token * D + reduce_offsets).to(tl.float32)
            reduce_square = (reduce_x * reduce_x).to(tl.bfloat16).to(tl.float32)
            component_sum += reduce_square
        lane_sum += component_sum
    square_sum = tl.sum(lane_sum, axis=0)

    pair_offsets = tl.arange(0, 4096)
    pair_mask = pair_offsets < D // 2
    even_offsets = pair_offsets * 2
    odd_offsets = even_offsets + 1
    row_base = token * D
    even = tl.load(x_ptr + row_base + even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
    odd = tl.load(x_ptr + row_base + odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)
    mean_square = (square_sum / D).to(tl.bfloat16).to(tl.float32)
    mean_with_eps = (mean_square + EPS).to(tl.bfloat16).to(tl.float32)
    rstd = tl.math.rsqrt(mean_with_eps).to(tl.bfloat16).to(tl.float32)
    even_weight = tl.load(weight_ptr + even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
    odd_weight = tl.load(weight_ptr + odd_offsets, mask=pair_mask, other=0.0).to(tl.float32)
    even_norm = ((even * rstd).to(tl.bfloat16).to(tl.float32) * even_weight).to(tl.bfloat16).to(tl.float32)
    odd_norm = ((odd * rstd).to(tl.bfloat16).to(tl.float32) * odd_weight).to(tl.bfloat16).to(tl.float32)

    freq_index = pair_offsets % 64
    position = tl.load(positions_ptr + token)
    freq_base = position * 128
    cos = tl.load(freqs_ptr + freq_base + freq_index, mask=pair_mask, other=0.0).to(tl.float32)
    sin = tl.load(freqs_ptr + freq_base + 64 + freq_index, mask=pair_mask, other=0.0).to(tl.float32)
    even_output = even_norm * cos - odd_norm * sin
    odd_output = odd_norm * cos + even_norm * sin
    tl.store(x_ptr + row_base + even_offsets, even_output.to(tl.bfloat16), mask=pair_mask)
    tl.store(x_ptr + row_base + odd_offsets, odd_output.to(tl.bfloat16), mask=pair_mask)


def fused_wan_qk_rmsnorm_rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    if q.shape != k.shape or q.ndim != 3 or q.shape[1:] != (40, 128):
        raise ValueError(f"Expected matching [tokens, 40, 128] Q/K, got q={q.shape}, k={k.shape}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("Fused Wan Q/K RMSNorm+RoPE requires BF16 Q/K")
    if q_weight.shape != (5120,) or k_weight.shape != (5120,):
        raise ValueError(f"Expected 5120-element Q/K weights, got {q_weight.shape} and {k_weight.shape}")
    if freqs.ndim != 2 or freqs.shape[1] != 128 or freqs.dtype != torch.float32:
        raise ValueError(f"Expected FP32 RoPE cache [rows, 128], got {freqs.shape} {freqs.dtype}")
    if positions.shape != (q.shape[0],) or positions.dtype != torch.int64:
        raise ValueError(f"Expected int64 RoPE positions [{q.shape[0]}], got {positions.shape} {positions.dtype}")

    with torch.cuda.device(q.device):
        torch.library.wrap_triton(_fused_wan_qk_rmsnorm_rope_kernel)[(q.shape[0], 2)](
            q,
            k,
            q_weight,
            k_weight,
            freqs,
            positions,
            D=5120,
            EPS=eps,
            num_warps=16,
        )


@torch.library.custom_op(
    "lightx2v::fused_wan_qk_rmsnorm_rope_",
    mutates_args=("q", "k"),
    device_types="cuda",
    schema="(Tensor(a!) q, Tensor(b!) k, Tensor q_weight, Tensor k_weight, Tensor freqs, Tensor positions, float eps) -> ()",
)
def _fused_wan_qk_rmsnorm_rope_custom_op(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> None:
    fused_wan_qk_rmsnorm_rope_(q, k, q_weight, k_weight, freqs, positions, eps)


@_fused_wan_qk_rmsnorm_rope_custom_op.register_fake
def _fused_wan_qk_rmsnorm_rope_fake(q, k, q_weight, k_weight, freqs, positions, eps) -> None:
    return None


def apply_fused_wan_qk_rmsnorm_rope_(q, k, q_weight, k_weight, freqs, positions, eps=1e-6) -> None:
    if torch.compiler.is_compiling():
        torch.ops.lightx2v.fused_wan_qk_rmsnorm_rope_(q, k, q_weight, k_weight, freqs, positions, eps)
    else:
        fused_wan_qk_rmsnorm_rope_(q, k, q_weight, k_weight, freqs, positions, eps)
