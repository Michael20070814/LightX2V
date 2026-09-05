import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from lightx2v_train.data.cache_dataset import CacheDataset, _single_sample_collate
from lightx2v_train.data.utils import load_video_tensor
from lightx2v_train.model_zoo.minimax_h3 import abot_action as actions
from lightx2v_train.model_zoo.minimax_h3.abot_dataset import SourceFrameSampler, pick_prompt, window_start
from lightx2v_train.model_zoo.minimax_h3.action_script import annotate, annotate_from_keys9
from lightx2v_train.model_zoo.minimax_h3.world_cache import MiniMaxH3WorldCacheModel
from lightx2v_train.model_zoo.minimax_h3.world_condition_encoder import MiniMaxH3WorldConditionEncoder
from lightx2v_train.model_zoo.minimax_h3.world_data_process import MiniMaxH3WorldProcessor
from lightx2v_train.model_zoo.minimax_h3.world_packing import WorldPackedSequenceBuilder
from lightx2v_train.model_zoo.minimax_h3.world_presentation import presentation_fl2va
from lightx2v_train.trainers.cache_build import CacheBuildTrainer, _to_cpu
from lightx2v_train.utils.registry import build_data, build_sample_processor
from torch.utils.data import DataLoader, Dataset


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}

    def convert_tokens_to_ids(self, token):
        return {"<|vision_start|>": 1000, "<|image_pad|>": 1001, "<|vision_end|>": 1002}[token]


class ImageProcessor:
    merge_size = 2

    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.zeros(16, 3), "image_grid_thw": torch.tensor([[1, 4, 4]])}


class TinyEncoder(MiniMaxH3WorldConditionEncoder):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.processor = SimpleNamespace(image_processor=ImageProcessor())
        self.action_embeddings = {}
        self.calls = []

    def _encode_ids(self, ids, pixel_values=None, grid=None):
        self.calls.append((ids.clone(), pixel_values is not None))
        return ids.float()[:, None].expand(-1, 4).clone()


class TinyDataset(Dataset):
    height = width = 64
    num_frames = 5
    skipped = [{"id": "missing", "reason": "video missing"}]

    def __init__(self):
        script = ["the man walks forward, camera follows him"] * 2
        self.record = {"prompt": "A room", "action_script": script}
        self.samples = [self.record]
        self.sample = {
            "inputs": {"video": torch.zeros(3, 5, 64, 64), "audio": torch.zeros(2, 6400), "first_frame": torch.zeros(3, 1, 64, 64)},
            "conditioning": copy.deepcopy(self.record),
            "meta": {"dataset_index": 0, "num_frames": 5, "source_fingerprint": "source-a"},
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return copy.deepcopy(self.sample)

    def cache_source_record(self, index):
        return copy.deepcopy(self.record)

    def cache_source_prompt(self, index):
        return self.record["prompt"]


class WorldCacheTests(unittest.TestCase):
    def test_vae_tiling_toggle(self):
        config = {"model": {"running_dtype": "bf16", "vae_tiling": False}, "training": {"method": "flow_matching"}}
        model = MiniMaxH3WorldCacheModel(config)
        model.patch_size = (1, 2, 2)
        model.video_latent_channels, model.audio_latent_channels, model.vae_spatial_scale_factor = 24, 32, 16
        model.video_vae = SimpleNamespace(enable_tiling=Mock(), disable_tiling=Mock())
        with patch("lightx2v_train.model_zoo.minimax_h3.minimax_h3_t2av.MiniMaxH3T2AVModel.load_components"):
            model.load_components(load_transformer=False, load_vae=True, load_condition_encoder=False)
        model.video_vae.disable_tiling.assert_called_once()
        model.video_vae.enable_tiling.assert_not_called()

    def test_temporal_sampling_and_pooling(self):
        self.assertEqual(actions.window_offsets(8), [0, 1, 2, 3, 5, 6, 7, 8])
        for frames in (5, 22, 124, 141):
            latent_t = actions.latent_t_for(frames)
            self.assertEqual(actions.frame_spans(latent_t)[-1][1], frames)
        matrix = np.zeros((124, 17), np.float32)
        matrix[3, 0] = 1
        matrix[:, 12] = 1
        pooled = actions.bin_to_latent(matrix, 37)
        self.assertEqual(pooled[0, 0], 0)
        self.assertEqual(pooled[1, 0], 1)
        self.assertEqual(pooled[0, 12], 0.25)
        self.assertEqual(pooled[1, 12], 1)
        self.assertEqual(len(annotate(pooled)), 37)
        start = window_start("sample", 0, 1800, actions.window_span(130))
        self.assertEqual(start, window_start("sample", 0, 1800, actions.window_span(130)))

    def test_conflicting_keys_and_scene_fallback(self):
        script = annotate_from_keys9(np.ones((2, 9), dtype=np.float32))
        self.assertEqual(script, ["the man stands still, camera holds steady"] * 2)
        self.assertEqual(pick_prompt({"scene_static": "!!!", "narrative": "A room"}), "A room")

    def test_first_frame_uses_processed_video_and_silence(self):
        frame = np.arange(80 * 40 * 3, dtype=np.uint8).reshape(40, 80, 3)
        reader = SimpleNamespace(get_meta_data=lambda: {"fps": 30}, count_frames=lambda: 200, get_data=lambda index: frame, close=lambda: None)
        with patch("lightx2v_train.data.utils.imageio.get_reader", return_value=reader):
            video = load_video_tensor("unused", 32, 32, SourceFrameSampler(actions.window_offsets(5)))
        expected = np.asarray(Image.fromarray(frame).resize((64, 32), Image.Resampling.BILINEAR).crop((16, 0, 48, 32)))
        actual = ((video[:, 0].permute(1, 2, 0) + 1) * 127.5).round().byte().numpy()
        np.testing.assert_array_equal(actual, expected)
        sample = {"inputs": {"video": video}, "conditioning": {"action_script": ["a", "b"]}, "meta": {"video_start_time": 0, "has_audio": False}}
        output = MiniMaxH3WorldProcessor({"model": {}})(sample)
        torch.testing.assert_close(output["inputs"]["first_frame"], output["inputs"]["video"][:, :1])
        self.assertEqual(tuple(output["inputs"]["audio"].shape), (2, 6400))
        self.assertEqual(output["inputs"]["audio"].count_nonzero(), 0)

    def test_independent_actions_and_visual_tags(self):
        encoder = TinyEncoder()
        script = ["the man walks forward", "the man stands still", "the man walks forward"]
        condition = encoder.encode_world("A room", Image.new("RGB", (64, 64)), script)
        self.assertEqual(len(encoder.calls), 3)
        self.assertTrue(encoder.calls[0][1])
        self.assertFalse(encoder.calls[1][1])
        spans = condition["action_text_spans"]
        self.assertEqual(spans[-1][1], condition["prompt_embeds"].shape[0])
        torch.testing.assert_close(condition["prompt_embeds"][slice(*spans[0])], condition["prompt_embeds"][slice(*spans[2])])
        _, head_tags = presentation_fl2va(encoder.tokenizer, "A room", [4])
        torch.testing.assert_close(condition["text_token_tags"][: len(head_tags)], head_tags)
        self.assertEqual(int((head_tags == 0).sum()), 6)
        encoder.encode_world("Another room", Image.new("RGB", (64, 64)), script)
        self.assertEqual(len(encoder.calls), 4)

    def test_packed_time_binding_and_precision(self):
        spans = [(20 + index * 10, 30 + index * 10) for index in range(37)]
        builder = WorldPackedSequenceBuilder()
        packed = builder._build_packed_fl2va(390, 37, 4, 6, 207, [0], action_text_spans=spans, pad_used_to=1088)
        positions = packed["img_position_ids"][0, :, 0]
        differences = []
        for index, (lo, hi) in enumerate(spans):
            video_row = packed["action_video_start"] + index * packed["action_frame_rows"]
            differences.append(positions[video_row] - positions[lo])
            torch.testing.assert_close(positions[lo:hi], positions[lo].expand(hi - lo))
        torch.testing.assert_close(torch.stack(differences), differences[0].expand(37))
        self.assertEqual(packed["action_real_used"], 390 + 6 + 414 + 37 * 6)
        converted = _to_cpu({"packed": packed}, torch.bfloat16)["packed"]
        self.assertEqual(converted["img_position_ids"].dtype, torch.float64)
        self.assertEqual(converted["cu_seqlens"].dtype, torch.int32)
        self.assertEqual(converted["action_text_rows"].dtype, torch.long)
        with self.assertRaises(ValueError):
            builder._build_packed_fl2va(390, 37, 4, 6, 207, [0], action_text_spans=spans[:-1])
        with self.assertRaises(ValueError):
            builder._build_packed_fl2va(390, 37, 4, 6, 207, [0], action_text_spans=spans, pad_used_to=20)

    def test_cache_roundtrip_resume_and_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "model": {"name": "minimax_h3_world", "running_dtype": "bf16", "pretrained_model_name_or_path": directory},
                "training": {"method": "flow_matching"},
                "cache_build": {"output_dir": directory, "save_dtype": "bf16", "seed": 0, "overwrite": False},
            }
            model = MiniMaxH3WorldCacheModel(config)
            model.condition_encoder = TinyEncoder()
            model.audio_latent_channels = 32
            model.encode_to_cache_latents = lambda sample: {"video_latents": torch.zeros(1, 24, 2, 4, 4), "audio_latents": torch.zeros(1, 16, 32)}
            model._encode_video_latents = lambda frame: torch.zeros(1, 24, 1, 4, 4)
            dataset = TinyDataset()
            loader = DataLoader(dataset, batch_size=1, collate_fn=_single_sample_collate)
            trainer = CacheBuildTrainer(config)
            trainer.set_model(model)
            trainer.set_data(loader)
            trainer.train()
            cached = CacheDataset([Path(directory) / "cache_data.jsonl"])[0]
            self.assertEqual(cached["conditioning"]["active"], "positive")
            self.assertNotIn("unconditional", cached["conditioning"])
            condition = cached["conditioning"]["positive"]
            packed = condition["packed"]
            self.assertEqual(packed["refiner_cu_seqlens"].tolist(), [0, *[lo for lo, _ in condition["action_text_spans"]], condition["prompt_embeds"].shape[0]])
            self.assertEqual(tuple(cached["inputs"]["audio_latents"].shape), (2, 32, 8))
            self.assertEqual(tuple(condition["keyframe_cond_anchor"].shape), (4, 96))
            self.assertEqual(packed["img_position_ids"].dtype, torch.float64)
            self.assertTrue((Path(directory) / "cache_skipped.jsonl").is_file())
            calls = len(model.condition_encoder.calls)
            trainer.train()
            self.assertEqual(len(model.condition_encoder.calls), calls)
            dataset.sample["meta"]["source_fingerprint"] = "changed"
            with self.assertRaisesRegex(ValueError, "source"):
                trainer.train()
            with self.assertRaisesRegex(ValueError, "unconditional"):
                CacheDataset([Path(directory) / "cache_data.jsonl"], prompt_dropout_rate=1)[0]
            meta = json.loads((Path(directory) / "cache_meta.json").read_text())
            self.assertEqual(meta["action_pad_used_to"], packed["seq_len"])

    def test_dataset_registry_and_missing_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory) / "data" / "ab" / "abcd"
            episode_dir.mkdir(parents=True)
            config = {"model": {"name": "minimax_h3_world"}, "data": {"train": {"name": "abot_dataset", "data_root": directory, "num_clips": 1, "num_workers": 0}}}
            processor = build_sample_processor(config)
            with self.assertRaisesRegex(ValueError, "No complete"):
                build_data(config, "train", processor)

    def test_dataset_complete_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory) / "data" / "ab" / "abcd"
            episode_dir.mkdir(parents=True)
            (episode_dir / "video.mp4").touch()
            (episode_dir / "annotations.tar").touch()
            episode = {
                "keys": np.zeros((180, 11), np.float32),
                "R_cw": np.tile(np.eye(3), (180, 1, 1)),
                "C_world": np.zeros((180, 3)),
                "fps": 30,
                "total_frames": 180,
                "control_scheme": "WASD_QE_locomotion_IJKL_rotation",
                "caption": {"narrative": "A room"},
            }
            reader = SimpleNamespace(get_meta_data=lambda: {"fps": 30}, count_frames=lambda: 180, get_data=lambda index: np.full((32, 64, 3), index, np.uint8), close=lambda: None)
            config = {
                "model": {"name": "minimax_h3_world"},
                "data": {"train": {"name": "abot_dataset", "data_root": directory, "num_clips": 1, "num_workers": 0, "height": 32, "width": 32, "num_frames": 5, "pin_memory": False}},
            }
            with (
                patch("lightx2v_train.model_zoo.minimax_h3.abot_dataset.actions.read_episode", return_value=episode),
                patch("lightx2v_train.model_zoo.minimax_h3.abot_dataset.imageio.get_reader", return_value=reader),
                patch("lightx2v_train.data.utils.imageio.get_reader", return_value=reader),
            ):
                loader = build_data(config, "train", build_sample_processor(config))
                first = next(iter(loader))
                second = next(iter(loader))
            self.assertEqual(len(loader.dataset), 1)
            torch.testing.assert_close(first["inputs"]["video"], second["inputs"]["video"])
            ids = first["meta"]["source_frame_ids"]
            self.assertEqual((ids - ids[0]).tolist(), actions.window_offsets(5))
            self.assertEqual(len(first["conditioning"]["action_script"]), 2)
            self.assertEqual(first["meta"]["dataset_index"], 0)

    def test_audio_decode_and_failure(self):
        processor = MiniMaxH3WorldProcessor({"model": {}, "data": {"processor": {"ffmpeg": "ffmpeg"}}})
        stereo = np.array([[0.25, -0.25], [0.5, -0.5]], dtype="<f4")
        with patch("subprocess.run", return_value=SimpleNamespace(stdout=stereo.tobytes())) as run:
            waveform = processor._load_video_audio("episode.mp4", 1.5, 4)
        torch.testing.assert_close(waveform, torch.tensor([[0.25, 0.5, 0, 0], [-0.25, -0.5, 0, 0]]))
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-ss") + 1], "1.5")
        with patch("subprocess.run", side_effect=OSError("decoder failed")):
            with self.assertRaises(OSError):
                processor._load_video_audio("episode.mp4", 0, 4)


if __name__ == "__main__":
    unittest.main()
