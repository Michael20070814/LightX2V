import copy
import json
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from lightx2v_train.data.utils import VideoFrameSelection, load_video_tensor
from lightx2v_train.infer.minimax_h3_world import MiniMaxH3WorldInferencer, world_inference_scheduler, write_world_video_audio
from lightx2v_train.model_zoo.minimax_h3.world_cache import encode_world_condition, patchify
from lightx2v_train.model_zoo.minimax_h3.world_infer_data import WorldInferenceDataset, action_script_for_request
from lightx2v_train.model_zoo.minimax_h3.world_inference import unpatchify_world_video
from lightx2v_train.model_zoo.minimax_h3.world_lora import read_world_lora
from lightx2v_train.model_zoo.minimax_h3.world_transformer import WorldTransformer
from lightx2v_train.utils.registry import build_data, build_inferencer, build_model
from safetensors.torch import save_file
from test_minimax_h3_world_training import add_adapter, tiny_model


def config_for(directory, lora_path=None):
    return {
        "model": {"name": "minimax_h3_world", "running_dtype": "fp32", "transformer_param_dtype": "fp32", "attention_backend": "sdpa", "pretrained_model_name_or_path": str(directory)},
        "scheduler": {"num_train_timesteps": 1000},
        "inference": {"method": "minimax_h3_world_infer", "output_dir": str(directory / "output"), "num_inference_steps": 2, "lora_config": {"path": str(lora_path)} if lora_path else {}},
    }


def write_adapter(path, transformer, reference=False, metadata=True):
    path.mkdir()
    state = {name: value.detach().contiguous() for name, value in transformer.named_parameters() if "lora_" in name}
    if reference:
        state = {key.replace("base.transformer_blocks.", "blocks.").replace(".o_proj.", ".out_proj."): value for key, value in state.items()}
    save_file(state, str(path / "pytorch_lora_weights.safetensors"))
    if metadata:
        (path / "adapter_config.json").write_text(json.dumps({"rank": 4, "alpha": 4, "format": "lightx2v_h3_world_v1", "target_modules": ["qkv_proj", "o_proj"], "qkv_layout": "head_qkv_dim"}))
    return path


class WorldInferenceTests(unittest.TestCase):
    def test_actions_follow_requested_latent_length(self):
        for frames, length in ((5, 2), (124, 37), (243, 72), (481, 142)):
            forward = action_script_for_request({"num_frames": frames, "action_preset": "forward"})
            self.assertEqual(len(forward), length)
            self.assertEqual(len(set(forward)), 1)
            sequence = action_script_for_request({"num_frames": frames, "action_keys": [["W"]] * (length - 1) + [["L", "F"]]})
            self.assertEqual(sequence[:-1], forward[:-1])
            self.assertNotEqual(sequence[-1], forward[-1])
        for request in (
            {"num_frames": 6, "action_preset": "forward"},
            {"num_frames": 5, "action_keys": [["W"]]},
            {"num_frames": 5, "action_keys": [["unknown"], []]},
            {"num_frames": 5, "action_preset": "forward", "action_keys": [[], []]},
        ):
            with self.assertRaises(ValueError):
                action_script_for_request(request)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.json"
            path.write_text(json.dumps([["W"], ["L", "F"]]))
            actual = action_script_for_request({"num_frames": 5, "action_keys_path": str(path)})
        self.assertNotEqual(actual[0], actual[1])

    def test_first_frame_matches_training_pixels(self):
        array = np.random.default_rng(1).integers(0, 256, (73, 111, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.fromarray(array).save(path)
            dataset = WorldInferenceDataset({"height": 32, "width": 64, "num_frames": 5, "samples": [{"first_frame": str(path), "prompt": "scene", "action_preset": "still"}]})
            reader = Mock()
            reader.get_data.return_value = array
            sampler = Mock()
            sampler.sample.return_value = VideoFrameSelection((0,), 0.0)
            with patch("lightx2v_train.data.utils.imageio.get_reader", return_value=reader):
                training = ((load_video_tensor("unused", 32, 64, sampler) + 1) * 0.5).clamp(0, 1)
            torch.testing.assert_close(dataset[0]["first_frame_tensor"], training, rtol=0, atol=0)

    def test_lora_native_and_reference_load_identical_factors(self):
        torch.manual_seed(22)
        base = tiny_model()
        reference = WorldTransformer(copy.deepcopy(base), "sdpa")
        add_adapter(reference)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base.save_pretrained(root / "transformer")
            for external in (False, True):
                path = write_adapter(root / str(external), reference, reference=external, metadata=not external)
                config = config_for(root, path if not external else path / "pytorch_lora_weights.safetensors")
                config["inference"]["lora_config"]["strength"] = 0.5
                with patch("torch.cuda.is_available", return_value=False):
                    model = build_model(config)
                    model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
                for name, value in reference.named_parameters():
                    expected = value * 0.5 if ".lora_B." in name else value
                    torch.testing.assert_close(dict(model.transformer.named_parameters())[name], expected, rtol=0, atol=0)
                self.assertTrue(all(not value.requires_grad for value in model.transformer.parameters()))

    def test_lora_rejects_old_layout_and_partial_pairs(self):
        transformer = WorldTransformer(tiny_model(), "sdpa")
        adapted = copy.deepcopy(transformer)
        add_adapter(adapted)
        with tempfile.TemporaryDirectory() as directory:
            path = write_adapter(Path(directory) / "adapter", adapted)
            config_path = path / "adapter_config.json"
            metadata = json.loads(config_path.read_text())
            metadata.pop("qkv_layout")
            config_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "QKV layout"):
                read_world_lora(path, transformer)
            state = {
                name.replace("base.transformer_blocks.", "blocks.").replace(".o_proj.", ".out_proj."): value.detach().contiguous() for name, value in adapted.named_parameters() if "lora_A" in name
            }
            weights = path / "partial.safetensors"
            save_file(state, str(weights))
            with self.assertRaisesRegex(ValueError, "Missing"):
                read_world_lora(weights, transformer)

    def test_lora_loading_shards_each_tp_rank_without_changing_factors(self):
        base = tiny_model()
        reference = WorldTransformer(copy.deepcopy(base), "sdpa")
        add_adapter(reference)
        full = dict(reference.named_parameters())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base.save_pretrained(root / "transformer")
            adapter = write_adapter(root / "adapter", reference)
            config = config_for(root, adapter)
            config["distributed"] = {"tensor_parallel": {"enabled": True, "size": 2}}
            for rank in range(2):
                with ExitStack() as stack:
                    stack.enter_context(patch("torch.cuda.is_available", return_value=False))
                    for module in ("runtime.tensor_parallel", "model_zoo.minimax_h3.world_transformer", "model_zoo.minimax_h3.world_inference"):
                        stack.enter_context(patch(f"lightx2v_train.{module}.tp_size", return_value=2))
                    for module in ("runtime.tensor_parallel", "model_zoo.minimax_h3.world_transformer"):
                        stack.enter_context(patch(f"lightx2v_train.{module}.tp_rank", return_value=rank))
                    stack.enter_context(patch("lightx2v_train.model_zoo.minimax_h3.world_inference.get_world_size", return_value=2))
                    model = build_model(config)
                    model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
                    for name, value in model.transformer.named_parameters():
                        if "lora_" not in name:
                            continue
                        expected = full[name.replace(".module.", ".")]
                        spec = getattr(value, "_tp_shard", None)
                        if spec:
                            self.assertEqual(spec[1], 1)
                            expected = expected.chunk(2, dim=spec[0])[rank]
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_invalid_inference_options_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            for options in ({"enable_cfg": True}, {"cfg_scale": 2}, {"num_inference_steps": 0}, {"video_flow_shift": float("nan")}, {"fps": 30}):
                config = config_for(Path(directory))
                config["inference"].update(options)
                model = build_model(config)
                with self.assertRaises(ValueError):
                    model.load_components(load_transformer=True, load_vae=True, load_condition_encoder=True)

    def test_schedule_moves_toward_clean_and_counts_model_calls(self):
        config = {"model": {"running_dtype": "fp32"}, "scheduler": {"num_train_timesteps": 1000}}
        with patch("torch.cuda.is_available", return_value=False):
            for shift in (12, 3):
                scheduler = world_inference_scheduler(config, 50, shift)
                self.assertEqual(len(scheduler.infer_sigmas), 51)
                x = torch.tensor([2.0])
                for step in range(50):
                    x = scheduler.step(torch.tensor([-3.0]), step, x)
                torch.testing.assert_close(x, torch.tensor([5.0]))
                self.assertEqual(float(scheduler.infer_sigmas[-1]), 0)

    def test_patch_roundtrip_preserves_channel_and_frame_order(self):
        video = torch.arange(24 * 7 * 4 * 6).reshape(1, 24, 7, 4, 6)
        torch.testing.assert_close(unpatchify_world_video(patchify(video), 7, 4, 6), video, rtol=0, atol=0)

    def test_registered_inference_runs_real_tiny_dit_and_shared_conditioning(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            root = Path(directory)
            tiny_model().save_pretrained(root / "transformer")
            image_path = root / "frame.png"
            Image.new("RGB", (32, 32), "red").save(image_path)
            config = config_for(root)
            config["data"] = {
                "val": {
                    "name": "minimax_h3_world_infer",
                    "num_frames": 5,
                    "height": 32,
                    "width": 32,
                    "num_workers": 0,
                    "samples": [{"first_frame": str(image_path), "prompt": "scene", "action_keys": [["W"], ["L"]]}],
                }
            }
            model = build_model(config)
            model.load_components(load_transformer=True, load_vae=False, load_condition_encoder=False)
            model._encode_video_latents = Mock(return_value=torch.zeros(1, 24, 1, 2, 2))
            model.condition_encoder = SimpleNamespace(
                encode_world=Mock(
                    return_value={
                        "prompt_embeds": torch.randn(24, 16),
                        "text_token_tags": torch.ones(24, dtype=torch.long),
                        "action_text_spans": [(4, 14), (14, 24)],
                    }
                )
            )
            dataloader = build_data(config, "val")
            request = dataloader.dataset[0]
            video, audio, condition = model.prepare_world_inference(request, 0)
            repeated = model.prepare_world_inference(request, 0)
            torch.testing.assert_close(video, repeated[0], rtol=0, atol=0)
            torch.testing.assert_close(audio, repeated[1], rtol=0, atol=0)
            self.assertEqual(condition["packed"]["refiner_cu_seqlens"].tolist(), [0, 4, 14, 24])
            self.assertEqual(condition["packed"]["action_text_spans_local"], [(4, 14), (14, 24)])
            training_condition = encode_world_condition(model, request["first_frame_tensor"], "scene", request["action_script"], 5, (2, 2), seed=0)
            torch.testing.assert_close(condition["keyframe_cond_anchor"], training_condition["keyframe_cond_anchor"], rtol=0, atol=0)
            model.decode_world_latents = Mock(return_value=(torch.zeros(5, 32, 32, 3), torch.zeros(2, 6667)))
            inferencer = build_inferencer(config)
            self.assertIsInstance(inferencer, MiniMaxH3WorldInferencer)
            inferencer.set_model(model)
            inferencer.set_data(dataloader)
            with patch("lightx2v_train.infer.minimax_h3_world.write_world_video_audio") as writer:
                paths = inferencer.infer()
            self.assertEqual(paths, [str(root / "output" / "00000.mp4")])
            writer.assert_called_once()
            final_video, final_audio, _ = model.decode_world_latents.call_args.args
            self.assertTrue(torch.isfinite(final_video).all() and torch.isfinite(final_audio).all())
            self.assertFalse(torch.equal(final_video, video))

    def test_decode_denormalizes_video_and_stereo_audio(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            model = build_model(config_for(Path(directory)))
        model.video_vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[2] * 24, latents_std=[3] * 24), decode=Mock(return_value=SimpleNamespace(sample=torch.zeros(1, 3, 5, 32, 32))))
        model.audio_vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[4] * 32, latents_std=[5] * 32), decode=Mock(return_value=SimpleNamespace(sample=torch.zeros(2, 1, 6400))))
        model._active_cache_encoder = lambda vae: nullcontext(vae)
        video, audio = model.decode_world_latents(torch.ones(1, 2, 96), torch.ones(1, 16, 32), {"num_frames": 5, "height": 32, "width": 32})
        self.assertEqual(video.shape, (5, 32, 32, 3))
        self.assertEqual(audio.shape, (2, 6400))
        self.assertTrue(torch.all(model.video_vae.decode.call_args.args[0] == 5))
        self.assertTrue(torch.all(model.audio_vae.decode.call_args.args[0] == 9))
        torch.testing.assert_close(video[0, 0, 0], torch.tensor([0.485, 0.456, 0.406]))

    def test_mux_writes_audio_and_requested_video_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            write_world_video_audio(torch.zeros(5, 32, 32, 3), torch.zeros(2, 6400), path)
            reader = imageio.get_reader(str(path))
            try:
                self.assertEqual(reader.count_frames(), 5)
                self.assertEqual(reader.get_meta_data()["fps"], 24)
                self.assertEqual(reader.get_meta_data()["audio_codec"], "aac")
            finally:
                reader.close()
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
