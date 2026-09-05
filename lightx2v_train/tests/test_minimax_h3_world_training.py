import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import MiniMaxH3Transformer3DModel
from lightx2v_train.model_capabilities import FlowMatchingSFTCapability
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_flow_matching_capability import MiniMaxH3FlowMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.world_cache import WorldCacheCapability
from lightx2v_train.model_zoo.minimax_h3.world_packing import WorldPackedSequenceBuilder
from lightx2v_train.model_zoo.minimax_h3.world_training import WorldFlowMatchingCapability, discrete_schedule, validate_world_sample
from lightx2v_train.model_zoo.minimax_h3.world_transformer import WorldTransformer, build_action_mask, load_world_weights
from lightx2v_train.model_zoo.native.minimax_h3 import audio_latent_num_frames
from lightx2v_train.runtime.distributed import cleanup_distributed, init_distributed
from lightx2v_train.runtime.tensor_parallel import clip_tensor_parallel_grad_norm_, gather_tensor, sync_tensor_parallel_gradients
from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler
from lightx2v_train.utils.registry import build_model
from peft import LoraConfig, inject_adapter_in_model


def tiny_model():
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=4,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=64,
        text_dim=16,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )


def tiny_sample():
    spans = [(4, 14), (14, 24)]
    audio_t = audio_latent_num_frames(5)
    packed = WorldPackedSequenceBuilder()._build_packed_fl2va(24, 2, 2, 2, audio_t, [0], action_text_spans=spans)
    packed = WorldPackedSequenceBuilder()._build_packed_fl2va(24, 2, 2, 2, audio_t, [0], action_text_spans=spans, pad_used_to=packed["seq_len"])
    packed["refiner_cu_seqlens"] = torch.tensor([0, 4, 14, 24], dtype=torch.int32)
    return {
        "meta": {"schema": "minimax_h3_world_v1", "num_frames": 5},
        "inputs": {"video_latents": torch.randn(1, 24, 2, 2, 2), "audio_latents": torch.randn(2, 32, audio_t)},
        "conditioning": {
            "positive": {
                "prompt_embeds": torch.randn(24, 16),
                "text_token_tags": torch.ones(24, dtype=torch.long),
                "keyframe_cond_anchor": torch.randn(1, 96),
                "imgvid_cond_noise_aug": 0.999,
                "action_text_spans": spans,
                "packed": packed,
            }
        },
    }


def add_adapter(model):
    model.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=4, lora_alpha=4, target_modules=r"base\.transformer_blocks\.\d+\.attn\.(qkv_proj|o_proj)"), model)
    # Exercise gradients of both factors, not only the initially zero B factor.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.01)


class WorldTrainingTests(unittest.TestCase):
    def test_fused_attention_matches_independent_projections(self):
        torch.manual_seed(9)
        base = tiny_model()
        original = base.transformer_blocks[0].attn
        fused = WorldTransformer(copy.deepcopy(base), "sdpa").base.transformer_blocks[0].attn
        hidden = torch.randn(1, 7, original.to_q.in_features)
        expected_qkv = torch.stack(
            [projection(hidden).unflatten(-1, (original.heads, original.head_dim)) for projection in (original.to_q, original.to_k, original.to_v)], dim=-2
        ).flatten(-3)
        torch.testing.assert_close(fused.qkv_proj(hidden), expected_qkv)
        from lightx2v_train.model_zoo.minimax_h3.world_transformer import WorldAttentionProcessor

        processor = WorldAttentionProcessor()
        torch.testing.assert_close(processor(fused, hidden), processor(original, hidden))

    def test_tp_streaming_loader_and_lora_use_contiguous_heads(self):
        torch.manual_seed(19)
        base = tiny_model()
        reference = WorldTransformer(copy.deepcopy(base), "sdpa")
        add_adapter(reference)
        hidden = torch.randn(1, 7, 32)
        expected = reference.base.transformer_blocks[0].attn.qkv_proj(hidden)
        with tempfile.TemporaryDirectory() as directory:
            base.save_pretrained(directory)
            for rank in range(2):
                with patch("lightx2v_train.runtime.tensor_parallel.tp_size", return_value=2), patch("lightx2v_train.runtime.tensor_parallel.tp_rank", return_value=rank), patch(
                    "lightx2v_train.model_zoo.minimax_h3.world_transformer.tp_size", return_value=2
                ), patch("lightx2v_train.model_zoo.minimax_h3.world_transformer.tp_rank", return_value=rank):
                    parallel = copy.deepcopy(reference)
                    parallel.shard()
                    projection = parallel.base.transformer_blocks[0].attn.qkv_proj
                    self.assertEqual(projection.module.lora_B["default"].weight._tp_shard, (0, 1))
                    torch.testing.assert_close(projection(hidden), expected.chunk(2, dim=-1)[rank])
                    with torch.device("meta"):
                        streamed = WorldTransformer(tiny_model(), "sdpa")
                    streamed.shard()
                    load_world_weights(streamed, directory, "cpu", torch.float32)
                    for index, block in enumerate(streamed.base.transformer_blocks):
                        full = reference.base.transformer_blocks[index].attn.qkv_proj.get_base_layer().weight
                        torch.testing.assert_close(block.attn.qkv_proj.module.weight, full.chunk(2, dim=0)[rank], rtol=0, atol=0)

    def test_tp_launch_validation_precedes_weight_loading(self):
        config = {
            "model": {"name": "minimax_h3_world", "running_dtype": "fp32"},
            "training": {"method": "flow_matching", "train_type": "lora"},
            "data": {"train": {"name": "cache_dataset"}},
            "distributed": {"tensor_parallel": {"enabled": True, "size": 2}},
        }
        with patch("torch.cuda.is_available", return_value=False):
            model = build_model(config)
        with self.assertRaisesRegex(ValueError, "torchrun"):
            model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)

    def test_world_and_t2av_remain_separate_model_paths(self):
        config = {"model": {"name": "minimax_h3_t2av", "running_dtype": "fp32"}, "training": {"method": "flow_matching", "train_type": "lora"}}
        with patch("torch.cuda.is_available", return_value=False):
            t2av = build_model(copy.deepcopy(config)).ensure_capabilities().require(FlowMatchingSFTCapability)
            config["model"]["name"] = "minimax_h3_world"
            world = build_model(copy.deepcopy(config)).ensure_capabilities().require(FlowMatchingSFTCapability)
            config["cache_build"] = {}
            cache = build_model(config).ensure_capabilities().require(FlowMatchingSFTCapability)
        self.assertIs(type(t2av), MiniMaxH3FlowMatchingCapability)
        self.assertEqual(t2av.video_shift, 6)
        self.assertEqual(t2av.audio_shift, 3)
        self.assertIs(type(world), WorldFlowMatchingCapability)
        self.assertIs(type(cache), WorldCacheCapability)

    def test_streaming_loader_matches_pretrained(self):
        torch.manual_seed(10)
        base = tiny_model()
        expected = WorldTransformer(copy.deepcopy(base), "sdpa")
        with tempfile.TemporaryDirectory() as directory:
            base.save_pretrained(directory)
            with torch.device("meta"):
                actual = WorldTransformer(tiny_model(), "sdpa")
            load_world_weights(actual, directory, "cpu", torch.float32)
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)

    def test_world_loss_uses_discrete_weight_and_framework_noise(self):
        sample = tiny_sample()
        observed = []

        def predict(video, audio, condition, vs, aus):
            observed.append((video, audio, vs, aus))
            return torch.zeros_like(video, requires_grad=True), torch.zeros_like(audio, requires_grad=True)

        config = {"model": {"running_dtype": "fp32"}, "scheduler": {"num_train_timesteps": 1000}}
        model = SimpleNamespace(config=config, device="cpu", denoiser_module=lambda: predict)
        capability = WorldFlowMatchingCapability(model)
        scheduler = RectifiedFlowMatchingScheduler(config)
        context = SimpleNamespace(noise_scheduler=scheduler)
        with patch("torch.randint", return_value=torch.tensor([200])), patch("torch.randn_like", side_effect=lambda x: torch.zeros_like(x)):
            result = capability.compute_loss(sample, context)
        sigma, weight = discrete_schedule()
        v, a = sample["inputs"].values()
        expected = (v.square().mean() + a.square().mean()) * weight[200]
        torch.testing.assert_close(result.loss, expected)
        torch.testing.assert_close(observed[0][0].square().mean(), v.square().mean() * (1 - sigma[200]).square())
        result.loss.backward()

    def test_discrete_distribution_and_weights(self):
        sigma, weights = discrete_schedule()
        base = torch.linspace(1, 0, 1001)[:-1]
        self.assertTrue(torch.all(sigma >= base))
        self.assertGreater(sigma.mean().item(), 0.6)
        self.assertEqual(float(sigma[0]), 1)
        self.assertGreater(float(sigma[-1]), 0)
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)
        self.assertEqual(float(weights[0]), 0)
        reference = torch.exp(-2 * ((sigma * 1000 - 500) / 1000) ** 2)
        reference = (reference - reference.min()) * (1000 / (reference - reference.min()).sum())
        torch.testing.assert_close(weights, reference)

    def test_visibility_all_row_pairs(self):
        sample = tiny_sample()
        packed = sample["conditioning"]["positive"]["packed"]
        mask = build_action_mask(packed, torch.device("cpu"), "sdpa")
        spans, start = packed["action_text_spans_local"], packed["action_video_start"]
        for q in range(packed["seq_len"]):
            for k in range(packed["seq_len"]):
                aq = next((i for i, (lo, hi) in enumerate(spans) if lo <= q < hi), -1)
                ak = next((i for i, (lo, hi) in enumerate(spans) if lo <= k < hi), -1)
                fq = q - start if start <= q < start + 2 else -1
                fk = k - start if start <= k < start + 2 else -1
                expected = q == k or (q < packed["action_real_used"] and k < packed["action_real_used"] and (ak < 0 or aq == ak or fq == ak) and not (aq >= 0 and fk >= 0 and aq != fk))
                self.assertEqual(bool(mask[q, k]), expected, (q, k))

    def test_cache_validation_rejects_corrupt_binding(self):
        sample = tiny_sample()
        sample["conditioning"]["positive"]["packed"]["action_text_spans_local"] = [[4, 14], [14, 24]]
        validate_world_sample(sample)
        sample["conditioning"]["positive"]["packed"]["action_video_start"] += 1
        with self.assertRaisesRegex(ValueError, "action_video_start"):
            validate_world_sample(sample)

    def test_lora_scope_and_checkpoint_backward(self):
        torch.manual_seed(11)
        model = WorldTransformer(tiny_model(), "sdpa")
        add_adapter(model)
        model.enable_gradient_checkpointing()
        condition = tiny_sample()["conditioning"]["positive"]
        result = model(torch.randn(1, 2, 96), torch.randn(1, 2 * audio_latent_num_frames(5), 32), condition, torch.tensor([0.7]), torch.tensor([0.5]))
        sum(x.square().mean() for x in result).backward()
        names = [name for name, p in model.named_parameters() if p.grad is not None]
        self.assertEqual(len(names), 8)
        self.assertTrue(all("transformer_blocks" in name and ("qkv_proj" in name or "o_proj" in name) and "lora_" in name for name in names))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_refiner_segments_do_not_mix_actions(self):
        torch.manual_seed(12)
        model = WorldTransformer(tiny_model(), "sdpa")
        sample = tiny_sample()["conditioning"]["positive"]
        inputs = []
        handle = model.base.transformer_blocks[0].register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
        video, audio = torch.randn(1, 2, 96), torch.randn(1, 2 * audio_latent_num_frames(5), 32)
        with torch.no_grad():
            model(video, audio, sample, torch.tensor([0.7]), torch.tensor([0.5]))
            sample["prompt_embeds"][4:14] += 3
            model(video, audio, sample, torch.tensor([0.7]), torch.tensor([0.5]))
        handle.remove()
        torch.testing.assert_close(inputs[0][:, :4], inputs[1][:, :4], rtol=0, atol=0)
        torch.testing.assert_close(inputs[0][:, 14:], inputs[1][:, 14:], rtol=0, atol=0)
        self.assertFalse(torch.equal(inputs[0][:, 4:14], inputs[1][:, 4:14]))


def distributed_probe():
    init_distributed({"distributed": {"tensor_parallel": {"enabled": True, "size": 2}}})
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(81)
    reference = WorldTransformer(tiny_model(), "sdpa")
    add_adapter(reference)
    parallel = copy.deepcopy(reference)
    parallel.shard()
    reference.to(device)
    parallel.to(device)
    parallel.enable_gradient_checkpointing()
    sample = tiny_sample()["conditioning"]["positive"]
    video = torch.randn(1, 2, 96, device=device)
    audio = torch.randn(1, 2 * audio_latent_num_frames(5), 32, device=device)
    sigmas = (torch.tensor([0.7], device=device), torch.tensor([0.5], device=device))
    expected = reference(video, audio, sample, *sigmas)
    actual = parallel(video, audio, sample, *sigmas)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    sum(x.square().mean() for x in expected).backward()
    sum(x.square().mean() for x in actual).backward()
    params = [p for p in parallel.parameters() if p.requires_grad]
    sync_tensor_parallel_gradients(params)
    expected_params = dict(reference.named_parameters())
    for name, p in parallel.named_parameters():
        if not p.requires_grad:
            continue
        grad = p.grad
        if hasattr(p, "_tp_shard"):
            grad = gather_tensor(grad, *p._tp_shard)
        torch.testing.assert_close(grad, expected_params[name.replace(".module.", ".")].grad, rtol=5e-4, atol=3e-6)
    norm = clip_tensor_parallel_grad_norm_(params, 0.1)
    expected_norm = torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], 0.1)
    torch.testing.assert_close(norm, expected_norm, rtol=2e-5, atol=2e-6)
    # Compare the compiled sparse mask to dense attention on the same sharded model.
    parallel.attention_backend = "flex"
    with torch.no_grad():
        sparse = parallel(video, audio, sample, *sigmas)
    for a, b in zip(sparse, expected):
        torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
    # A different action span at the same packed length must not reuse a stale mask.
    changed = copy.deepcopy(sample)
    packed = changed["packed"]
    spans = [(4, 16), (16, 24)]
    changed["action_text_spans"] = spans
    changed["packed"] = WorldPackedSequenceBuilder()._build_packed_fl2va(
        24,
        2,
        2,
        2,
        audio_latent_num_frames(5),
        [0],
        action_text_spans=spans,
        pad_used_to=packed["seq_len"],
    )
    changed["packed"]["refiner_cu_seqlens"] = torch.tensor([0, 4, 16, 24], dtype=torch.int32)
    reference.zero_grad(set_to_none=True)
    parallel.zero_grad(set_to_none=True)
    expected = reference(video, audio, changed, *sigmas)
    sparse = parallel(video, audio, changed, *sigmas)
    for a, b in zip(sparse, expected):
        torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
    sum(x.square().mean() for x in expected).backward()
    sum(x.square().mean() for x in sparse).backward()
    sync_tensor_parallel_gradients(params)
    for name, p in parallel.named_parameters():
        if p.requires_grad:
            grad = gather_tensor(p.grad, *p._tp_shard) if hasattr(p, "_tp_shard") else p.grad
            torch.testing.assert_close(grad, expected_params[name.replace(".module.", ".")].grad, rtol=8e-4, atol=4e-6)
    # More than Dynamo's default recompile budget, with one fixed padding budget.
    with torch.no_grad():
        for extra in range(10):
            variant = copy.deepcopy(sample)
            spans = [(4, 14 + extra), (14 + extra, 24 + extra)]
            variant["prompt_embeds"] = torch.cat([sample["prompt_embeds"], sample["prompt_embeds"][:extra]])
            variant["packed"] = WorldPackedSequenceBuilder()._build_packed_fl2va(
                24 + extra,
                2,
                2,
                2,
                audio_latent_num_frames(5),
                [0],
                action_text_spans=spans,
                pad_used_to=packed["seq_len"],
            )
            variant["packed"]["refiner_cu_seqlens"] = torch.tensor([0, 4, 14 + extra, 24 + extra], dtype=torch.int32)
            for a, b in zip(parallel(video, audio, variant, *sigmas), reference(video, audio, variant, *sigmas)):
                torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
    print("PASS: TP=2 outputs, gradients, clipping, FlexAttention backward and 10 variable-length masks match TP=1.", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        distributed_probe()
    else:
        unittest.main()
