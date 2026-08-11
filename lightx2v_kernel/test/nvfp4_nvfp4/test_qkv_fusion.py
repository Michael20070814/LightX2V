import pytest
import torch

from lightx2v_kernel.gemm import (
    cutlass_scaled_nvfp4_mm,
    cutlass_scaled_nvfp4_qkv_mm,
    scaled_nvfp4_quant,
)


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


@pytest.mark.parametrize("bias_enabled", [False, True])
@torch.inference_mode()
def test_qkv_fusion_matches_three_gemms(bias_enabled):
    m, k = 129, 256
    output_sizes = (128, 128, 128)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weights = [
        torch.randn((output_size, k), dtype=torch.bfloat16, device="cuda")
        for output_size in output_sizes
    ]
    biases = [
        torch.randn((output_size,), dtype=torch.bfloat16, device="cuda") if bias_enabled else None
        for output_size in output_sizes
    ]

    activation_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / activation.abs().max()
    ).float()
    activation_fp4, activation_scale = scaled_nvfp4_quant(
        activation,
        activation_global_scale,
    )

    weight_fp4 = []
    weight_scales = []
    alphas = []
    expected = []
    for weight, bias in zip(weights, biases):
        weight_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / weight.abs().max()
        ).float()
        alpha = 1.0 / (activation_global_scale * weight_global_scale)
        quantized_weight, weight_scale = scaled_nvfp4_quant(weight, weight_global_scale)
        weight_fp4.append(quantized_weight)
        weight_scales.append(weight_scale)
        alphas.append(alpha)
        expected.append(
            cutlass_scaled_nvfp4_mm(
                activation_fp4,
                quantized_weight,
                activation_scale,
                weight_scale,
                alpha,
                bias,
            )
        )

    actual = cutlass_scaled_nvfp4_qkv_mm(
        activation_fp4,
        torch.stack(weight_fp4, dim=0),
        activation_scale,
        torch.stack(weight_scales, dim=0),
        torch.stack(alphas, dim=0),
        None if not bias_enabled else torch.stack(biases, dim=0),
    )

    torch.testing.assert_close(
        actual,
        torch.cat(expected, dim=-1),
        atol=1e-1,
        rtol=1e-1,
    )
