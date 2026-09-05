"""ABot audio preparation and first-frame extraction using the H3 processor."""

import subprocess

import numpy as np
import torch

from lightx2v_train.model_zoo.native.minimax_h3 import audio_latent_num_frames, video_latent_num_frames
from lightx2v_train.utils.registry import SAMPLE_PROCESSOR_REGISTER

from .data_process import MiniMaxH3T2AVProcessor


class MiniMaxH3WorldProcessor(MiniMaxH3T2AVProcessor):
    requires_audio = False

    def __init__(self, config):
        super().__init__(config)
        self.ffmpeg = config.get("data", {}).get("processor", {}).get("ffmpeg")

    def __call__(self, sample):
        video = sample["inputs"]["video"]
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError("H3 world video must be [3,F,H,W].")
        frames = int(video.shape[1])
        latent_t = video_latent_num_frames(frames)
        if len(sample["conditioning"]["action_script"]) != latent_t:
            raise ValueError("Expected one action sentence for each video latent frame.")
        target_samples = audio_latent_num_frames(frames) * self.audio_hop_length
        meta = sample["meta"]
        waveform = torch.zeros(2, target_samples, dtype=torch.float32)
        if meta.get("has_audio", False):
            waveform = self._load_video_audio(meta["video_path"], meta["video_start_time"], target_samples)
        sample["inputs"]["video"] = ((video + 1.0) * 0.5).clamp_(0.0, 1.0)
        sample["inputs"]["audio"] = waveform
        sample["inputs"]["first_frame"] = sample["inputs"]["video"][:, :1].clone()
        meta.update(
            num_frames=frames,
            target_height=int(video.shape[-2]),
            target_width=int(video.shape[-1]),
            audio_sample_rate=self.audio_sample_rate,
            audio_start_sample=round(meta["video_start_time"] * self.audio_sample_rate),
        )
        return sample

    def _load_video_audio(self, path, start, target_samples):
        import imageio_ffmpeg

        executable = self.ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-ss",
                str(start),
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-t",
                str(target_samples / self.audio_sample_rate),
                "-ac",
                "2",
                "-ar",
                str(self.audio_sample_rate),
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
        samples = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2).copy()
        return self._slice_audio(torch.from_numpy(samples).T, 0, target_samples)


@SAMPLE_PROCESSOR_REGISTER("minimax_h3_world")
def build_world_processor(config):
    return MiniMaxH3WorldProcessor(config)
