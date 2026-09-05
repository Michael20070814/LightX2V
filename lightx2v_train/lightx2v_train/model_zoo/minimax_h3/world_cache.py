"""Positive-only FL2VA world-model cache adapter for Flow Matching SFT."""

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from lightx2v_train.model_capabilities import FlowMatchingSFTCapability
from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.model_zoo.native.minimax_h3 import audio_latent_num_frames, video_latent_num_frames

from .minimax_h3_t2av import MiniMaxH3T2AVModel
from .world_condition_encoder import MiniMaxH3WorldConditionEncoder
from .world_packing import WorldPackedSequenceBuilder
from .world_presentation import image_token_counts, presentation_fl2va, presentation_t2va


def patchify(video):
    batch, channels, frames, height, width = video.shape
    if batch != 1 or height % 2 or width % 2:
        raise ValueError("Expected one video with even latent spatial dimensions.")
    return video.reshape(1, channels, frames, height // 2, 2, width // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, channels * 4)


def tensor_image(frame):
    pixels = frame[:, 0].detach().float().cpu().clamp(0, 1)
    array = pixels.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def encode_world_condition(model, first_frame, prompt, script, num_frames, latent_hw, *, seed, pad_used_to=None):
    latent_t = video_latent_num_frames(num_frames)
    if len(script) != latent_t:
        raise ValueError("Expected one action sentence per video latent frame.")
    first_latent = model._encode_video_latents(first_frame.unsqueeze(0))
    if first_latent.shape != (1, 24, 1, *latent_hw):
        raise ValueError("The first-frame VAE output must match the target latent geometry.")
    anchor = patchify(first_latent).to(model.running_dtype)
    augmentation = float(model.config["model"].get("imgvid_cond_noise_aug", 0.999))
    if not 0 <= augmentation <= 1:
        raise ValueError("imgvid_cond_noise_aug must lie in [0,1].")
    if augmentation != 1:
        # The reference draws T+1 frames and uses its first latent as anchor noise.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, 24, latent_t + 1, *latent_hw), generator=generator, dtype=model.running_dtype)
        noise_rows = patchify(noise[:, :, :1]).to(anchor.device)
        amount = anchor.new_tensor(augmentation)
        anchor = amount * anchor + (1 - amount) * noise_rows
    condition = model.condition_encoder.encode_world(prompt, tensor_image(first_frame), script)
    spans = condition["action_text_spans"]
    packed = WorldPackedSequenceBuilder()._build_packed_fl2va(
        condition["prompt_embeds"].shape[0], latent_t, *latent_hw,
        audio_latent_num_frames(num_frames), [0], action_text_spans=spans, pad_used_to=pad_used_to,
    )
    packed["token_tags"][packed["text_pos"]] = condition["text_token_tags"]
    packed["refiner_cu_seqlens"] = torch.tensor([0, spans[0][0], *[hi for _, hi in spans]], dtype=torch.int32)
    condition.update(
        packed=packed, keyframe_cond_anchor=anchor.contiguous(), keyframe_indices=[0],
        imgvid_cond_noise_aug=augmentation, action_script=list(script), action_pad_used_to=packed["seq_len"],
    )
    return condition


class MiniMaxH3WorldCacheModel(MiniMaxH3T2AVModel):
    condition_encoder_cls = MiniMaxH3WorldConditionEncoder

    def register_capabilities(self):
        BaseModel.register_capabilities(self)
        self.capabilities.register(FlowMatchingSFTCapability, WorldCacheCapability(self))

    def load_components(self, *, load_transformer, load_vae, load_condition_encoder):
        if load_transformer:
            raise NotImplementedError("minimax_h3_world currently builds caches only; its SFT consumer is a separate task.")
        if self.config.get("training", {}).get("method", "flow_matching") != "flow_matching":
            raise ValueError("The world cache contract targets Flow Matching SFT.")
        for split in ("train", "val"):
            if self.config.get("data", {}).get(split, {}).get("prompt_dropout_rate", 0):
                raise ValueError("World caches contain positive conditions only; disable prompt dropout.")
        super().load_components(load_transformer=False, load_vae=load_vae, load_condition_encoder=load_condition_encoder)
        if self.patch_size != (1, 2, 2):
            raise ValueError("FL2VA action packing requires patch_size=(1,2,2).")
        if (self.video_latent_channels, self.audio_latent_channels, self.vae_spatial_scale_factor) != (24, 32, 16):
            raise ValueError("FL2VA world packing requires 24 video channels, 32 audio channels and spatial scale 16.")
        if self.video_vae is not None:
            if self.config["model"].get("vae_tiling", True):
                self.video_vae.enable_tiling()
            else:
                self.video_vae.disable_tiling()


class WorldCacheCapability(FlowMatchingSFTCapability):
    def __init__(self, model):
        self.model = model
        self.builder = WorldPackedSequenceBuilder()
        self.pad_used_to = None
        self.cache_metadata = {}

    def compute_loss(self, batch, context):
        raise NotImplementedError("This adapter provides the world-model SFT cache contract only.")

    def prepare_cache_dataset(self, dataset):
        if not all(hasattr(dataset, field) for field in ("height", "width", "num_frames", "cache_source_record")):
            raise ValueError("World cache construction requires a fixed-shape ABot dataset.")
        encoder = self.model.condition_encoder
        _, _, counts = image_token_counts(encoder.processor, [Image.new("RGB", (dataset.width, dataset.height))])
        latent_t = video_latent_num_frames(dataset.num_frames)
        frame_rows = (dataset.height // 32) * (dataset.width // 32)
        target_rows = (latent_t + 1) * frame_rows + 2 * audio_latent_num_frames(dataset.num_frames)
        lengths, used = {}, []
        for index in range(len(dataset)):
            record = dataset.cache_source_record(index)
            script = record["action_script"]
            if len(script) != latent_t:
                raise ValueError("Action script length does not match the latent timeline.")
            head_ids, _ = presentation_fl2va(encoder.tokenizer, record["prompt"], counts)
            for sentence in script:
                if sentence not in lengths:
                    lengths[sentence] = len(presentation_t2va(encoder.tokenizer, sentence)[0])
            used.append(len(head_ids) + sum(lengths[sentence] for sentence in script) + target_rows)
        maximum = max(used)
        configured = self.model.config["model"].get("action_pad_used_to")
        self.pad_used_to = ((maximum + 63) // 64) * 64 if configured is None else int(configured)
        if self.pad_used_to < maximum or self.pad_used_to % 64:
            raise ValueError(f"action_pad_used_to must be a multiple of 64 and at least {maximum}.")
        config = self.model.config
        model_config = config["model"]
        root = Path(model_config["pretrained_model_name_or_path"]).expanduser().resolve()
        components = (
            model_config.get("text_encoder_subfolder", "text_encoder"),
            model_config.get("processor_subfolder", "processor"),
            model_config.get("tokenizer_subfolder", "tokenizer"),
            self.model._component_subfolder(root, model_config.get("video_vae_subfolder"), ("vae", "video_vae")),
            model_config.get("audio_vae_subfolder", "audio_vae"),
        )
        files = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns] for component in components for p in sorted((root / component).rglob("*")) if p.is_file()}
        signature = {
            "schema": "minimax_h3_world_v1",
            "model": model_config,
            "weights": files,
            "padding": self.pad_used_to,
            "cache_seed": config["cache_build"]["seed"],
            "save_dtype": config["cache_build"]["save_dtype"],
            "processor": config.get("data", {}).get("processor", {}),
        }
        self.fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.cache_metadata = {
            "schema": signature["schema"],
            "cache_fingerprint": self.fingerprint,
            "action_pad_used_to": self.pad_used_to,
            "max_real_used": maximum,
            "height": dataset.height,
            "width": dataset.width,
            "num_frames": dataset.num_frames,
            "conditioning_roles": ["positive"],
            "training_method": "flow_matching",
        }

    @torch.inference_mode()
    def encode_training_cache(self, sample):
        if self.pad_used_to is None:
            raise RuntimeError("Prepare the complete cache dataset before encoding samples.")
        model = self.model
        latent_t = video_latent_num_frames(sample["meta"]["num_frames"])
        encoded = model.encode_to_cache_latents(sample)
        video = encoded["video_latents"]
        audio = encoded["audio_latents"].reshape(2, -1, model.audio_latent_channels).transpose(1, 2).contiguous()
        expected_audio = audio_latent_num_frames(sample["meta"]["num_frames"])
        if video.shape[2] != latent_t or audio.shape[-1] != expected_audio:
            raise ValueError("VAE output lengths do not match the selected video/action timeline.")
        anchor_seed = int(model.config["model"].get("condition_seed", 42))
        prompt = sample["conditioning"]["prompt"]
        condition = encode_world_condition(
            model, sample["inputs"]["first_frame"], prompt, sample["conditioning"]["action_script"],
            sample["meta"]["num_frames"], video.shape[-2:], seed=anchor_seed, pad_used_to=self.pad_used_to,
        )
        return {
            "inputs": {"video_latents": video, "audio_latents": audio},
            "conditioning": {"prompt": prompt, "positive": condition},
            "meta": {**sample["meta"], "schema": "minimax_h3_world_v1", "cache_fingerprint": self.fingerprint, "condition_seed": anchor_seed},
        }

    def validate_training_cache(self, cache, source_meta):
        meta = cache["meta"]
        if meta.get("cache_fingerprint") != self.fingerprint:
            raise ValueError("World cache configuration or weights changed; rebuild with --overwrite.")
        if meta.get("source_fingerprint") != source_meta.get("source_fingerprint"):
            raise ValueError("World cache source video, annotations, or window changed; rebuild with --overwrite.")
        condition = cache["conditioning"]["positive"]
        packed = condition["packed"]
        if packed["seq_len"] != self.pad_used_to:
            raise ValueError("World cache padding does not match the dataset scan.")
        latent_t = video_latent_num_frames(int(meta["num_frames"]))
        if len(packed["action_text_spans_local"]) != latent_t:
            raise ValueError("World cache is missing action-time bindings.")
        video, audio = cache["inputs"]["video_latents"], cache["inputs"]["audio_latents"]
        if video.ndim != 5 or video.shape[:3] != (1, 24, latent_t):
            raise ValueError("World cache has invalid video latent dimensions.")
        if audio.shape != (2, 32, audio_latent_num_frames(int(meta["num_frames"]))):
            raise ValueError("World cache has invalid audio latent dimensions.")
        frame_rows = (video.shape[-2] // 2) * (video.shape[-1] // 2)
        if condition["keyframe_cond_anchor"].shape != (frame_rows, 96):
            raise ValueError("World cache has invalid first-frame anchor dimensions.")
        text_len = condition["prompt_embeds"].shape[0]
        spans = packed["action_text_spans_local"]
        cursor = spans[0][0]
        if cursor <= 0:
            raise ValueError("World cache is missing the FL2VA head.")
        for lo, hi in spans:
            if lo != cursor or hi <= lo:
                raise ValueError("World action spans must be nonempty and contiguous.")
            cursor = hi
        if cursor != text_len or condition["text_token_tags"].shape != (text_len,):
            raise ValueError("World action spans and tags must cover all text rows.")
        expected_boundaries = torch.tensor([0, spans[0][0], *[hi for _, hi in spans]], dtype=torch.int32)
        if not torch.equal(packed["refiner_cu_seqlens"].cpu(), expected_boundaries):
            raise ValueError("World cache has invalid refiner segmentation.")
