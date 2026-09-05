"""H3 world-model cache producer and TP-capable SFT model."""

import json
import math
import random
from pathlib import Path

import torch
from loguru import logger
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file, save_file

from lightx2v_train.model_capabilities import FlowMatchingSFTCapability
from lightx2v_train.model_zoo.base import BaseModel
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class, resolve_transformer_dir
from lightx2v_train.runtime.distributed import is_main_process
from lightx2v_train.runtime.tensor_parallel import gather_tensor, shard_tensor, tp_size
from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import is_cache_build

from .world_cache import MiniMaxH3WorldCacheModel, WorldCacheCapability
from .world_inference import WorldInferenceMixin
from .world_training import WorldFlowMatchingCapability
from .world_transformer import WorldTransformer, load_world_weights


@MODEL_REGISTER("minimax_h3_world")
class MiniMaxH3WorldModel(WorldInferenceMixin, MiniMaxH3WorldCacheModel):
    def __init__(self, config):
        super().__init__(config)
        self.is_world_inference = config.get("inference", {}).get("method") == "minimax_h3_world_infer" and not is_cache_build(config)
        if not is_cache_build(config) and not self.is_world_inference:
            for key in ("max_train_iters", "save_every_iters"):
                if config["training"].get(key) is not None:
                    config["training"][key] = int(config["training"][key])
            lora = config["training"].setdefault("lora", {})
            lora.setdefault("rank", 32)
            lora.setdefault("alpha", 32)

    def register_capabilities(self):
        BaseModel.register_capabilities(self)
        capability = WorldCacheCapability if is_cache_build(self.config) else WorldFlowMatchingCapability
        self.capabilities.register(FlowMatchingSFTCapability, capability(self))

    def load_components(self, *, load_transformer, load_vae, load_condition_encoder):
        if load_transformer and not self.is_world_inference:
            if load_vae or load_condition_encoder or self.config["data"]["train"]["name"] != "cache_dataset":
                raise ValueError("World SFT consumes prebuilt training data caches; build them with cache_data.py first.")
            if self.config["training"].get("train_type") != "lora":
                raise ValueError("World SFT currently supports train_type=lora only.")
            tp = self.config.get("distributed", {}).get("tensor_parallel", {})
            if tp.get("enabled", False) and tp_size() != int(tp.get("size", 2)):
                raise ValueError("Initialize the configured tensor parallel group with torchrun before loading the world model.")
        if self.is_world_inference:
            self.validate_inference_config()
        super().load_components(load_transformer=False, load_vae=load_vae, load_condition_encoder=load_condition_encoder)
        if not load_transformer:
            return
        backend = self.config["model"].get("attention_backend", "flex")
        if backend not in {"flex", "sdpa"}:
            raise ValueError("World attention_backend must be flex (training) or sdpa (small validation).")
        self._world_directory = resolve_transformer_dir(self.pretrained_model_path)
        cls = _transformer_class()
        with torch.device("meta"):
            base = cls.from_config(cls.load_config(str(self._world_directory)))
            self.transformer = WorldTransformer(base, backend)
        self._world_loaded = False
        if self.is_world_inference:
            self._world_load_device = "cpu" if self.config["inference"].get("transformer_cpu_offload", True) else self.device
            lora_path = (self.config["inference"].get("lora_config") or {}).get("path")
            if lora_path:
                self.load_lora_for_infer(lora_path)
            else:
                self.transformer.requires_grad_(False)
                self.transformer.shard()
                load_world_weights(self.transformer, self._world_directory, self._world_load_device, self.transformer_param_dtype)
                self._world_loaded = True
            self.transformer.eval()

    def add_lora(self, rank, alpha, target_modules):
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if target_modules and set(target_modules) != {"qkv_proj", "o_proj"}:
            raise ValueError("World LoRA targets are qkv_proj and o_proj in the DiT backbone only.")
        self.transformer.requires_grad_(False)
        targets = r"base\.transformer_blocks\.\d+\.attn\.(qkv_proj|o_proj)"
        inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, init_lora_weights="gaussian"), self.transformer)
        self._world_lora_config = {"rank": rank, "alpha": alpha, "target_modules": ["qkv_proj", "o_proj"], "format": "lightx2v_h3_world_v1", "qkv_layout": "head_qkv_dim"}
        # Configure sharding before materializing any large pretrained parameters.
        self.transformer.shard()
        seed = int(self.config.get("training", {}).get("seed", 42))
        torch.manual_seed(seed)
        random.seed(seed)
        load_world_weights(self.transformer, self._world_directory, getattr(self, "_world_load_device", self.device), self.transformer_param_dtype)
        self._world_loaded = True

    def load_lora_for_infer(self, lora_path, adapter_name=None):
        if self._world_loaded:
            raise ValueError("World LoRA must be configured before loading/sharding the model; create a new model to change adapters.")
        if adapter_name not in {None, "default"}:
            raise ValueError("World inference supports one default adapter.")
        from .world_lora import read_world_lora

        options = self.config["inference"].get("lora_config") or {}
        strength = float(options.get("strength", 1.0))
        if not math.isfinite(strength):
            raise ValueError("World LoRA strength must be finite.")
        state, rank, alpha = read_world_lora(lora_path, self.transformer, alpha=options.get("alpha"))
        self.add_lora(rank, alpha, ["qkv_proj", "o_proj"])
        with torch.no_grad():
            for name, parameter in self.transformer.named_parameters():
                key = name.replace(".module.", ".")
                if key not in state:
                    continue
                value = state.pop(key)
                spec = getattr(parameter, "_tp_shard", None)
                if spec:
                    value = shard_tensor(value, *spec)
                parameter.copy_(value * strength if ".lora_B." in key else value)
        if state:
            raise ValueError(f"Unloaded world LoRA keys: {sorted(state)}")
        self.transformer.requires_grad_(False)
        self._infer_lora_adapter_name = "default"
        logger.info("Loaded world inference LoRA from {} rank={} alpha={} strength={}", lora_path, rank, alpha, strength)

    def apply_tensor_parallel(self):
        expected = int(self.config["distributed"]["tensor_parallel"].get("size", 2))
        if tp_size() != expected or not self._world_loaded:
            raise ValueError(f"World training requires initialized TP={expected}; launch with torchrun.")

    def log_model_structure(self):
        trainable = sum(p.numel() for p in self.trainable_parameters())
        total = sum(p.numel() for p in self.transformer.parameters())
        logger.info("[world] TP={} local_parameters={} local_trainable={} lora={}", tp_size(), total, trainable, self._world_lora_config)

    def save_lora_weights(self, save_dir, **kwargs):
        state = {}
        for name, param in self.transformer.named_parameters():
            if not param.requires_grad:
                continue
            value = param.detach()
            spec = getattr(param, "_tp_shard", None)
            if spec:
                value = gather_tensor(value, *spec)
            if is_main_process():
                state[name.replace(".module.", ".")] = value.cpu().contiguous()
        if is_main_process():
            path = Path(save_dir)
            save_file(state, str(path / "pytorch_lora_weights.safetensors"), metadata={"format": "lightx2v_h3_world_v1"})
            (path / "adapter_config.json").write_text(json.dumps(self._world_lora_config, indent=2) + "\n")

    def load_lora_weights_for_resume(self, lora_path, **kwargs):
        path = Path(lora_path)
        if json.loads((path / "adapter_config.json").read_text()) != self._world_lora_config:
            raise ValueError("World LoRA rank, alpha, format or QKV layout differs from the checkpoint.")
        state = load_file(str(path / "pytorch_lora_weights.safetensors"))
        with torch.no_grad():
            for name, param in self.transformer.named_parameters():
                if not param.requires_grad:
                    continue
                value = state.pop(name.replace(".module.", "."))
                spec = getattr(param, "_tp_shard", None)
                if spec:
                    value = shard_tensor(value, *spec)
                if value.shape != param.shape:
                    raise ValueError(f"Wrong LoRA checkpoint shape for {name}.")
                param.copy_(value)
        if state:
            raise ValueError(f"Unexpected world LoRA checkpoint keys: {sorted(state)}")
