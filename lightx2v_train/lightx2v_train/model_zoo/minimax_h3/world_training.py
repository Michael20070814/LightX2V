"""Discrete, weighted Flow Matching SFT on positive-only H3 world caches."""

import math

import torch
import torch.nn.functional as F

from lightx2v_train.model_capabilities import LossResult
from lightx2v_train.runtime.tensor_parallel import broadcast_tensor_parallel_value

from ..native.minimax_h3 import audio_latent_num_frames, video_latent_num_frames
from .world_cache import WorldCacheCapability, patchify
from .world_packing import WorldPackedSequenceBuilder


def discrete_schedule(shift=2.22):
    if not math.isfinite(shift) or shift <= 1:
        raise ValueError("World SFT flow shift must be finite and >1 to favor high noise.")
    base = torch.linspace(1, 0, 1001, dtype=torch.float32)[:-1]
    sigmas = shift * base / (1 + (shift - 1) * base)
    curve = torch.exp(-2 * (sigmas - 0.5).square())
    weights = curve - curve.min()
    weights *= 1000 / weights.sum()
    return sigmas, weights


def validate_world_sample(sample):
    if sample["meta"].get("schema") != "minimax_h3_world_v1":
        raise ValueError("World SFT requires minimax_h3_world_v1 training data caches.")
    if sample["conditioning"].get("active", "positive") != "positive":
        raise ValueError("World SFT requires positive-only conditioning.")
    condition = sample["conditioning"]["positive"]
    video, audio = (sample["inputs"][name] for name in ("video_latents", "audio_latents"))
    frames = int(sample["meta"]["num_frames"])
    latent_t, audio_t = video_latent_num_frames(frames), audio_latent_num_frames(frames)
    if video.ndim != 5 or video.shape[:3] != (1, 24, latent_t) or any(x % 2 for x in video.shape[-2:]):
        raise ValueError("Invalid world video latent geometry.")
    if audio.shape != (2, 32, audio_t):
        raise ValueError("Invalid world stereo audio latent geometry.")
    frame_rows = video.shape[-2] * video.shape[-1] // 4
    if condition["keyframe_cond_anchor"].shape != (frame_rows, 96):
        raise ValueError("World cache must contain exactly one first-frame anchor.")
    augmentation = float(condition["imgvid_cond_noise_aug"])
    if not math.isfinite(augmentation) or not 0 <= augmentation <= 1:
        raise ValueError("Invalid cached first-frame noise augmentation.")
    embeds, tags = condition["prompt_embeds"], condition["text_token_tags"]
    if embeds.ndim != 2 or tags.shape != (embeds.shape[0],) or not ((tags >= 0) & (tags <= 2)).all():
        raise ValueError("Invalid cached world text embeddings/tags.")
    spans = condition["action_text_spans"]
    if len(spans) != latent_t or int(spans[0][0]) <= 0:
        raise ValueError("World cache requires a head and one action span per latent frame.")
    cursor = int(spans[0][0])
    for lo, hi in spans:
        if int(lo) != cursor or int(hi) <= cursor:
            raise ValueError("World action spans must be contiguous and nonempty.")
        cursor = int(hi)
    if cursor != embeds.shape[0]:
        raise ValueError("World action spans must cover the text tail.")
    packed = condition["packed"]
    expected = WorldPackedSequenceBuilder()._build_packed_fl2va(
        embeds.shape[0],
        latent_t,
        *video.shape[-2:],
        audio_t,
        [0],
        action_text_spans=spans,
        pad_used_to=int(packed["seq_len"]),
    )
    expected["token_tags"][expected["text_pos"]] = tags.cpu()
    expected["refiner_cu_seqlens"] = torch.tensor([0, spans[0][0], *[hi for _, hi in spans]], dtype=torch.int32)
    for key, value in expected.items():
        actual = packed.get(key)
        if isinstance(value, (list, tuple)) and isinstance(actual, (list, tuple)):
            equal = torch.equal(torch.as_tensor(value), torch.as_tensor(actual))
        elif torch.is_tensor(value):
            equal = torch.is_tensor(actual) and torch.equal(value, actual.cpu())
        else:
            equal = value == actual
        if not equal:
            raise ValueError(f"Invalid world packing field {key}; rebuild the training cache.")
    return video, audio, condition


class WorldFlowMatchingCapability(WorldCacheCapability):
    def __init__(self, model):
        super().__init__(model)
        options = model.config["model"].get("capabilities", {}).get("flow_matching", {})
        self.video_schedule = discrete_schedule(float(options.get("video_flow_shift", 2.22)))
        self.audio_schedule = discrete_schedule(float(options.get("audio_flow_shift", 2.22)))
        self.video_weight = float(options.get("video_loss_weight", 1))
        self.audio_weight = float(options.get("audio_loss_weight", 1))
        if any(not math.isfinite(w) or w < 0 for w in (self.video_weight, self.audio_weight)) or self.video_weight + self.audio_weight == 0:
            raise ValueError("World modality loss weights must be finite, nonnegative and not both zero.")

    def compute_loss(self, batch, context):
        scheduler = context.noise_scheduler
        if scheduler.num_train_timesteps != 1000 or scheduler.do_time_shift:
            raise ValueError("World SFT requires 1000 discrete steps and scheduler time shifting disabled.")
        with torch.no_grad():
            video, audio, condition = validate_world_sample(batch)
            video = patchify(video).unsqueeze(0).to(device=self.model.device, dtype=torch.float32)
            audio = audio.permute(0, 2, 1).reshape(1, -1, 32).to(device=self.model.device, dtype=torch.float32)
            index = broadcast_tensor_parallel_value(torch.randint(0, 1000, (1,), device=self.model.device))
            video_sigma, video_weight = (table.to(self.model.device)[index] for table in self.video_schedule)
            audio_sigma, audio_weight = (table.to(self.model.device)[index] for table in self.audio_schedule)
            video_noise = broadcast_tensor_parallel_value(torch.randn_like(video))
            audio_noise = broadcast_tensor_parallel_value(torch.randn_like(audio))
            noisy_video = scheduler.add_noise(video, video_noise, video_sigma)
            noisy_audio = scheduler.add_noise(audio, audio_noise, audio_sigma)
            video_target = scheduler.build_train_gt(video, video_noise)
            audio_target = scheduler.build_train_gt(audio, audio_noise)
        predicted_video, predicted_audio = self.model.denoiser_module()(noisy_video, noisy_audio, condition, video_sigma, audio_sigma)
        # H3's DiT predicts clean-noise; the framework target is noise-clean.
        video_loss = F.mse_loss(-predicted_video.float(), video_target)
        audio_loss = F.mse_loss(-predicted_audio.float(), audio_target)
        loss = self.video_weight * video_weight.squeeze() * video_loss + self.audio_weight * audio_weight.squeeze() * audio_loss
        return LossResult(
            loss=loss,
            metrics={
                "video_loss": video_loss.detach(),
                "audio_loss": audio_loss.detach(),
                "video_sigma": video_sigma.squeeze(),
                "audio_sigma": audio_sigma.squeeze(),
                "video_timestep_weight": video_weight.squeeze(),
                "audio_timestep_weight": audio_weight.squeeze(),
            },
        )
