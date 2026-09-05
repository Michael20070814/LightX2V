"""Inference operations on the same world-model modules used for SFT."""

import math
from contextlib import contextmanager

import torch

from lightx2v_train.model_zoo.native.minimax_h3 import audio_latent_num_frames, video_latent_num_frames
from lightx2v_train.runtime.distributed import get_world_size
from lightx2v_train.runtime.tensor_parallel import tp_size

from .world_cache import encode_world_condition, patchify


def unpatchify_world_video(rows, frames, height, width):
    return rows.reshape(1, frames, height // 2, width // 2, 24, 2, 2).permute(0, 4, 1, 2, 5, 3, 6).reshape(1, 24, frames, height, width)


class WorldInferenceMixin:
    def validate_inference_config(self):
        options = self.config["inference"]
        if options.get("enable_cfg", False) or float(options.get("cfg_guidance_scale", options.get("cfg_scale", 1))) != 1:
            raise ValueError("World inference currently supports positive conditioning only (CFG disabled, scale=1).")
        for key, default in (("video_flow_shift", 12), ("audio_flow_shift", 3)):
            value = float(options.get(key, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"inference.{key} must be finite and positive.")
        steps = options.get("num_inference_steps", 50)
        if isinstance(steps, bool) or int(steps) != steps or steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer.")
        if float(options.get("fps", 24)) != 24:
            raise ValueError("World inference requires fps=24 for action/audio alignment.")
        distributed = self.config.get("distributed", {})
        tp = distributed.get("tensor_parallel", {})
        expected = int(tp.get("size", 2)) if tp.get("enabled", False) else 1
        if tp_size() != expected or get_world_size() != expected:
            raise ValueError("World inference requires single-process execution or torchrun with TP=WORLD_SIZE.")
        if distributed.get("fsdp2", {}).get("enabled", False) or distributed.get("sequence_parallel", {}).get("enabled", False):
            raise ValueError("World inference supports tensor parallelism only; disable FSDP2 and SP.")
        if self.config["model"].get("checkpoint_path"):
            raise ValueError("Use inference.lora_config.path for world inference checkpoints.")

    @torch.no_grad()
    def prepare_world_inference(self, request, seed):
        frames, height, width = (request[key] for key in ("num_frames", "height", "width"))
        latent_t, audio_t = video_latent_num_frames(frames), audio_latent_num_frames(frames)
        latent_hw = (height // 16, width // 16)
        condition = encode_world_condition(
            self,
            request["first_frame_tensor"],
            request["prompt"],
            request["action_script"],
            frames,
            latent_hw,
            seed=int(self.config["model"].get("condition_seed", seed)),
            pad_used_to=self.config["model"].get("action_pad_used_to"),
        )
        # H3-World resets the CPU generator to the same seed for each modality.
        video = torch.randn((1, 24, latent_t, *latent_hw), generator=torch.Generator(device="cpu").manual_seed(seed), dtype=self.running_dtype)
        audio = torch.randn((2, 32, audio_t), generator=torch.Generator(device="cpu").manual_seed(seed), dtype=self.running_dtype)
        video = patchify(video).unsqueeze(0).to(self.device, dtype=self.latent_dtype)
        audio = audio.transpose(1, 2).reshape(1, 2 * audio_t, 32).to(self.device, dtype=self.latent_dtype)
        return video, audio, condition

    @contextmanager
    def inference_denoiser(self):
        self.transformer.to(self.device).eval()
        try:
            yield self.transformer
        finally:
            if self.config["inference"].get("transformer_cpu_offload", True) and self.device.type != "cpu":
                self.transformer.to("cpu")
                torch.cuda.empty_cache()

    @torch.no_grad()
    def decode_world_latents(self, video_rows, audio_rows, request):
        frames, height, width = (request[key] for key in ("num_frames", "height", "width"))
        video = unpatchify_world_video(video_rows.float(), video_latent_num_frames(frames), height // 16, width // 16)
        mean, std = self._latent_statistics(self.video_vae, video, (1, -1, 1, 1, 1))
        with self._active_cache_encoder(self.video_vae) as vae:
            pixels = vae.decode(video * std + mean).sample.float()
        pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        pixel_std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        if pixels.shape != (1, 3, frames, height, width):
            raise ValueError(f"Unexpected decoded video shape: {tuple(pixels.shape)}.")
        pixels = (pixels * pixel_std + pixel_mean).clamp_(0, 1)[0].permute(1, 2, 3, 0).cpu()
        audio = audio_rows.float().reshape(2, -1, 32).transpose(1, 2)
        mean, std = self._latent_statistics(self.audio_vae, audio, (1, -1, 1))
        with self._active_cache_encoder(self.audio_vae) as vae:
            waveform = vae.decode(audio * std + mean).sample
        if waveform.ndim != 3 or waveform.shape[:2] != (2, 1):
            raise ValueError(f"Unexpected decoded audio shape: {tuple(waveform.shape)}.")
        return pixels, waveform[:, 0].float().clamp(-1, 1).cpu()
