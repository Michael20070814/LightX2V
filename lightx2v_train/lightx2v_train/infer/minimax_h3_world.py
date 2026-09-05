"""Joint video/audio inference with the SFT world transformer."""

import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import torch
from diffusers.utils import export_to_video
from loguru import logger

from lightx2v_train.model_zoo.minimax_h3.world_transformer import build_action_mask
from lightx2v_train.runtime.distributed import barrier, is_main_process
from lightx2v_train.runtime.tensor_parallel import broadcast_tensor_parallel_value
from lightx2v_train.schedulers.flow_matching import RectifiedFlowMatchingScheduler
from lightx2v_train.utils.registry import INFERENCER_REGISTER

from .base import BaseInferencer


def world_inference_scheduler(config, steps, shift):
    scheduler = RectifiedFlowMatchingScheduler(config)
    base = torch.linspace(1, 0, steps + 1, dtype=torch.float32)[:-1]
    sigmas = shift * base / (1 + (shift - 1) * base)
    scheduler.set_timesteps(steps, sigmas=sigmas.tolist())
    return scheduler


def write_world_video_audio(video, audio, path, *, sample_rate=32000, ffmpeg=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".h3-infer-", dir=path.parent) as directory:
        root = Path(directory)
        video_path, audio_path, output = root / "video.mp4", root / "audio.wav", root / "output.mp4"
        export_to_video(video.numpy(), str(video_path), fps=24, macro_block_size=1)
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            pcm = audio.clamp(-1, 1).T.mul(32767).round().to(torch.int16).contiguous().numpy().astype("<i2")
            handle.writeframes(pcm.tobytes())
        subprocess.run(
            [
                ffmpeg or imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-af",
                "apad",
                "-t",
                str(len(video) / 24),
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        output.replace(path)


@INFERENCER_REGISTER("minimax_h3_world_infer")
class MiniMaxH3WorldInferencer(BaseInferencer):
    @torch.no_grad()
    def infer(self):
        self.model.validate_inference_config()
        if not self.output_infer_dir:
            raise ValueError("World inference requires inference.output_dir.")
        steps = int(self.infer_config.get("num_inference_steps", 50))
        video_scheduler = world_inference_scheduler(self.config, steps, float(self.infer_config.get("video_flow_shift", 12)))
        audio_scheduler = world_inference_scheduler(self.config, steps, float(self.infer_config.get("audio_flow_shift", 3)))
        saved = []
        for index, request in enumerate(self.dataloader_val):
            seed = int(request.get("seed", int(self.infer_config.get("seed", 0)) + index))
            logger.info("[world-infer] sample={} seed={} frames={} size={}x{}", index, seed, request["num_frames"], request["height"], request["width"])
            video, audio, condition = self.model.prepare_world_inference(request, seed)
            video = broadcast_tensor_parallel_value(video)
            audio = broadcast_tensor_parallel_value(audio)
            # One mask per request; it is independent of the denoising timestep.
            condition["attention_mask"] = build_action_mask(condition["packed"], self.model.device, self.model.transformer.attention_backend)
            with self.model.inference_denoiser() as denoiser:
                for step in range(steps):
                    with self.model.transformer_forward_context():
                        video_velocity, audio_velocity = denoiser(
                            video,
                            audio,
                            condition,
                            video_scheduler.infer_sigmas[step],
                            audio_scheduler.infer_sigmas[step],
                        )
                    # The DiT predicts clean-minus-noise; the common scheduler expects noise-minus-clean.
                    video = video_scheduler.step(-video_velocity.float(), step, video.float())
                    audio = audio_scheduler.step(-audio_velocity.float(), step, audio.float())
                    if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
                        logger.info("[world-infer] sample={} step={}/{}", index, step + 1, steps)
            del condition
            if is_main_process():
                pixels, waveform = self.model.decode_world_latents(video, audio, request)
                path = Path(self.output_infer_dir) / f"{index:05d}.mp4"
                write_world_video_audio(pixels, waveform, path, sample_rate=self.model.audio_sampling_rate, ffmpeg=self.infer_config.get("ffmpeg"))
                saved.append(str(path))
                logger.info("[world-infer] saved {}", path)
            barrier()
        return saved
