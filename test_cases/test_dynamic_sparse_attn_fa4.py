import pytest
import torch

from lightx2v.common.ops.attn.utils.sla_util import (
    block_lut_to_ordinal_metadata,
    centered_mean_pool,
    get_block_lut,
    get_block_map,
    mean_pool,
)
from lightx2v.common.ops.attn.utils.sparge_util import block_map_ordinal_lut_triton


def test_block_lut_to_ordinal_metadata_sorts_and_pads():
    lut = torch.tensor([[[[5, 1, 3], [4, 0, 2]]]])

    full_block_idx, full_block_cnt = block_lut_to_ordinal_metadata(lut, num_k_blocks=6)

    expected_idx = torch.tensor([[[[1, 3, 5, 0, 0, 0], [0, 2, 4, 0, 0, 0]]]], dtype=torch.int32)
    expected_cnt = torch.tensor([[[3, 3]]], dtype=torch.int32)
    torch.testing.assert_close(full_block_idx, expected_idx, atol=0, rtol=0)
    torch.testing.assert_close(full_block_cnt, expected_cnt, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton mean pooling requires CUDA")
def test_fa4_lut_matches_existing_map_path_for_gqa_and_partial_blocks():
    torch.manual_seed(42)
    q = torch.randn((1, 4, 257, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 259, 64), device="cuda", dtype=torch.bfloat16)

    k_mean = torch.mean(k, dim=-2, keepdim=True)
    old_pooled_k = mean_pool(k - k_mean, 128)
    new_pooled_k = centered_mean_pool(k, k_mean, 128)
    torch.testing.assert_close(new_pooled_k, old_pooled_k, atol=0, rtol=0)

    sparse_map, old_lut, old_topk = get_block_map(q, k, topk_ratio=0.67, BLKQ=128, BLKK=128)
    new_lut, new_topk, num_k_blocks = get_block_lut(q, k, topk_ratio=0.67, BLKQ=128, BLKK=128)

    torch.testing.assert_close(
        torch.sort(new_lut, dim=-1).values,
        torch.sort(old_lut, dim=-1).values,
        atol=0,
        rtol=0,
    )
    assert new_topk == old_topk
    assert num_k_blocks == 3

    old_idx, old_cnt = block_map_ordinal_lut_triton(sparse_map)
    new_idx, new_cnt = block_lut_to_ordinal_metadata(new_lut, num_k_blocks)
    torch.testing.assert_close(new_idx, old_idx, atol=0, rtol=0)
    torch.testing.assert_close(new_cnt, old_cnt, atol=0, rtol=0)
