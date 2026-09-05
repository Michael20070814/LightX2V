"""H3-specific ABot windows, reusing LightX2V's common video data utilities."""

import hashlib
import json
import random
import tarfile
from collections import OrderedDict
from pathlib import Path

import imageio.v2 as imageio
import torch
from loguru import logger

from lightx2v_train.data.cache_dataset import _single_sample_collate
from lightx2v_train.data.utils import VideoFrameSelection, load_video_tensor
from lightx2v_train.data.video_dataset import VideoDataset, _build_dataloader
from lightx2v_train.model_zoo.minimax_h3 import abot_action as actions
from lightx2v_train.model_zoo.minimax_h3.action_script import annotate
from lightx2v_train.model_zoo.native.minimax_h3 import video_latent_num_frames
from lightx2v_train.utils.registry import DATA_REGISTER


def window_start(sample_id, window, total_frames, span, seed=20260817):
    """Match H3-World's seeded slot order and deterministic per-window jitter."""
    usable = total_frames - span
    if usable < 0:
        raise ValueError(f"Episode has {total_frames} frames, needs {span}.")
    slots = list(range(max(1, total_frames // span)))
    random.Random(f"{sample_id}:{seed}").shuffle(slots)
    slot = slots[window % len(slots)]
    jitter = random.Random(f"{sample_id}:{window}:jit").randrange(0, max(1, span // 2))
    return min(usable, slot * span + jitter)


def pick_prompt(caption):
    scene = (caption.get("scene_static") or "").strip()
    return scene if len(scene.split()) >= 30 else (caption.get("narrative") or "").strip() or scene


class SourceFrameSampler:
    def __init__(self, frame_ids, source_fps=30.0):
        self.frame_ids = tuple(frame_ids)
        self.source_fps = source_fps

    def sample(self, reader):
        fps = float(reader.get_meta_data()["fps"])
        if abs(fps - self.source_fps) > 0.01:
            raise ValueError(f"ABot source video must be {self.source_fps}fps, got {fps}.")
        if self.frame_ids[-1] >= reader.count_frames():
            raise ValueError("Video is shorter than its selected annotation window.")
        return VideoFrameSelection(self.frame_ids, self.frame_ids[0] / self.source_fps)


class ABotDataset(VideoDataset):
    collate_fn = staticmethod(_single_sample_collate)

    def __init__(self, config, sample_processor):
        self.sample_processor = sample_processor
        self.root = Path(config["data_root"]).expanduser().resolve()
        self.height, self.width = int(config.get("height", 480)), int(config.get("width", 832))
        self.num_frames = int(config.get("num_frames", 124))
        latent_t = video_latent_num_frames(self.num_frames)
        if min(self.height, self.width) <= 0 or self.height % 32 or self.width % 32:
            raise ValueError("ABot height and width must be positive multiples of 32.")
        if config.get("random_start", False) or float(config.get("frame_rate", 24)) != 24:
            raise ValueError("ABot requires deterministic windows at 24fps.")
        if config.get("prompt_dropout_rate", 0) or config.get("dataset_repeat", 1) != 1:
            raise ValueError("ABot cache construction requires positive conditions and dataset_repeat=1.")
        seed = int(config.get("window_seed", 20260817))
        # The reference reserves six extra output frames when selecting windows.
        # Direct decoding needs no encoded-video padding, but preserves those starts.
        span = actions.window_span(self.num_frames + 6)
        episodes = sorted(path for path in (self.root / "data").glob("*/*") if path.is_dir())
        random.Random(seed).shuffle(episodes)
        if not episodes:
            raise ValueError(f"No ABot episodes found under {self.root}.")
        count = int(config.get("num_clips", len(episodes)))
        if count <= 0:
            raise ValueError("num_clips must be positive.")
        self.samples, self.skipped = [], []
        loaded = OrderedDict()
        for index in range(count):
            directory = episodes[index % len(episodes)]
            window = index // len(episodes)
            video, annotations = directory / "video.mp4", directory / "annotations.tar"
            try:
                if not video.is_file() or not annotations.is_file():
                    raise FileNotFoundError("video.mp4 or annotations.tar is missing")
                if directory not in loaded:
                    episode = actions.read_episode(str(annotations))
                    if episode["control_scheme"] != "WASD_QE_locomotion_IJKL_rotation":
                        raise ValueError(f"Unsupported control scheme: {episode['control_scheme']}")
                    if abs(episode["fps"] - 30.0) > 0.01:
                        raise ValueError("Reference ABot frame selection requires 30fps source annotations.")
                    reader = imageio.get_reader(str(video))
                    try:
                        info = reader.get_meta_data()
                        if abs(float(info["fps"]) - 30.0) > 0.01:
                            raise ValueError("Source video is not 30fps.")
                        if reader.count_frames() < episode["total_frames"]:
                            raise ValueError("Source video is shorter than its annotations.")
                    finally:
                        reader.close()
                    loaded[directory] = episode, actions.episode_translation_scale(episode), info
                    if len(loaded) > 8:
                        loaded.popitem(last=False)
                loaded.move_to_end(directory)
                episode, scale, info = loaded[directory]
                start = window_start(directory.name, window, episode["total_frames"], span, seed)
                matrix = actions.window_action_matrix(episode, start, self.num_frames, scale)
                script = annotate(actions.bin_to_latent(matrix, latent_t))
                prompt = pick_prompt(episode["caption"])
                if not prompt:
                    raise ValueError("Scene and narrative descriptions are both empty.")
                source_ids = [start + offset for offset in actions.window_offsets(self.num_frames)]
                record = {
                    "id": f"{directory.name}_w{window:03d}",
                    "video": str(video),
                    "annotations": str(annotations),
                    "prompt": prompt,
                    "source_frame_ids": source_ids,
                    "action_script": script,
                    "height": self.height,
                    "width": self.width,
                    "num_frames": self.num_frames,
                    "source_fps": 30,
                    "fps": 24,
                    "window_seed": seed,
                    "has_audio": bool(info.get("audio_codec")),
                    "source_files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in (video, annotations)},
                }
                record["source_fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
                self.samples.append({"prompt": prompt, "_original_record": record})
            except (ValueError, OSError, KeyError, RuntimeError, tarfile.TarError) as error:
                self.skipped.append({"id": directory.name, "window": window, "reason": str(error)})
                logger.warning("Skipping ABot {} window {}: {}", directory.name, window, error)
        if not self.samples:
            raise ValueError("No complete ABot samples remain; check data_root and num_clips.")
        self.dataset_repeat = 1
        logger.info("ABot selected {} windows; skipped {}.", len(self.samples), len(self.skipped))

    def __getitem__(self, index):
        record = self.cache_source_record(index)
        video = load_video_tensor(record["video"], self.height, self.width, SourceFrameSampler(record["source_frame_ids"]))
        sample = {
            "inputs": {"video": video},
            "conditioning": {"prompt": record["prompt"], "action_script": record["action_script"]},
            "meta": {
                "dataset_index": index,
                "video_path": record["video"],
                "video_start_time": record["source_frame_ids"][0] / 30.0,
                "source_frame_ids": torch.tensor(record["source_frame_ids"], dtype=torch.long),
                "source_fingerprint": record["source_fingerprint"],
                "has_audio": record["has_audio"],
                "sample_id": record["id"],
            },
        }
        return self.sample_processor(sample)


@DATA_REGISTER("abot_dataset")
def build_abot_dataset(data_config, train_or_val="train", sample_processor=None, unconditional_prompt=" "):
    if sample_processor is None:
        raise ValueError("abot_dataset requires the minimax_h3_world processor.")
    return _build_dataloader(ABotDataset(data_config, sample_processor), data_config, train_or_val)
