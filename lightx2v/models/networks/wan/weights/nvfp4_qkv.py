import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v_kernel.gemm import cutlass_scaled_nvfp4_qkv_mm, scaled_nvfp4_quant
from lightx2v_platform.base.global_var import AI_DEVICE


class WanNVFP4FusedQKV(WeightModule):
    """Load separate Wan Q/K/V checkpoints into one NVFP4 projection."""

    _PROJECTIONS = ("q", "k", "v")

    def __init__(self, block_prefix):
        super().__init__()
        self.block_prefix = block_prefix
        self.weight_names = tuple(f"{block_prefix}.{name}.weight" for name in self._PROJECTIONS)
        self.weight_scale_names = tuple(f"{block_prefix}.{name}.weight_scale" for name in self._PROJECTIONS)
        self.input_global_scale_names = tuple(f"{block_prefix}.{name}.input_global_scale" for name in self._PROJECTIONS)
        self.alpha_names = tuple(f"{block_prefix}.{name}.alpha" for name in self._PROJECTIONS)
        self.bias_names = tuple(f"{block_prefix}.{name}.bias" for name in self._PROJECTIONS)
        self.output_splits = None

    @staticmethod
    def _require_tensors(weight_dict, names):
        missing = [name for name in names if name not in weight_dict]
        if missing:
            raise KeyError(f"Missing fused QKV checkpoint tensors: {missing}")
        return [weight_dict[name] for name in names]

    def load(self, weight_dict):
        weights = self._require_tensors(weight_dict, self.weight_names)
        weight_scales = self._require_tensors(weight_dict, self.weight_scale_names)
        input_global_scales = self._require_tensors(weight_dict, self.input_global_scale_names)
        alpha_values = self._require_tensors(weight_dict, self.alpha_names)

        packed_k = weights[0].shape[1]
        if any(weight.ndim != 2 or weight.shape[1] != packed_k for weight in weights):
            raise ValueError(f"Fused QKV weights must share packed K; got {[tuple(weight.shape) for weight in weights]}")

        reference_input_scale = input_global_scales[0].reshape([])
        if any(not torch.equal(reference_input_scale, scale.reshape([])) for scale in input_global_scales[1:]):
            values = [scale.item() for scale in input_global_scales]
            raise ValueError(f"Fused QKV requires identical input_global_scale values; got {values}")

        self.output_splits = tuple(weight.shape[0] for weight in weights)
        if any(output_size % 128 != 0 for output_size in self.output_splits):
            raise ValueError(
                "Fused QKV requires every projection output size to be divisible by 128 "
                "because checkpoint scales are already swizzled"
            )
        if any(scale.ndim != 2 or scale.shape[0] != output_size for scale, output_size in zip(weight_scales, self.output_splits)):
            raise ValueError(
                f"Fused QKV weight scales must align with output rows; got "
                f"{[tuple(scale.shape) for scale in weight_scales]} for {self.output_splits}"
            )

        biases = [weight_dict.get(name) for name in self.bias_names]
        if any(bias is None for bias in biases) and not all(bias is None for bias in biases):
            raise ValueError("Fused QKV requires either all three biases or no biases")

        self.weight = torch.cat(weights, dim=0).contiguous()
        self.weight_scale = torch.cat(weight_scales, dim=0).contiguous()
        self.input_global_scale = reference_input_scale.contiguous()
        self.alpha_values = torch.stack([alpha.reshape([]) for alpha in alpha_values]).to(torch.float32)
        self.alpha = torch.cat(
            [alpha.expand(output_size) for alpha, output_size in zip(self.alpha_values, self.output_splits)],
            dim=0,
        ).contiguous()
        self.bias = None if all(bias is None for bias in biases) else torch.cat(biases, dim=0).contiguous()

    def apply(self, input_tensor):
        input_quant, input_scale = scaled_nvfp4_quant(input_tensor, self.input_global_scale)
        return cutlass_scaled_nvfp4_qkv_mm(
            input_quant,
            self.weight,
            input_scale,
            self.weight_scale,
            self.alpha,
            self.bias,
        )

    def state_dict(self, destination=None):
        if destination is None:
            destination = {}
        weight_parts = self.weight.split(self.output_splits, dim=0)
        scale_parts = self.weight_scale.split(self.output_splits, dim=0)
        bias_parts = (None,) * 3 if self.bias is None else self.bias.split(self.output_splits, dim=0)
        for index in range(3):
            destination[self.weight_names[index]] = weight_parts[index]
            destination[self.weight_scale_names[index]] = scale_parts[index]
            destination[self.input_global_scale_names[index]] = self.input_global_scale
            destination[self.alpha_names[index]] = self.alpha_values[index]
            if bias_parts[index] is not None:
                destination[self.bias_names[index]] = bias_parts[index]
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        del block_index, adapter_block_index
        self.load(destination)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        del block_index, adapter_block_index
        raise NotImplementedError("NVFP4 fused QKV does not support lazy loading")

    def _move_to(self, device, non_blocking=False):
        for name in ("weight", "weight_scale", "input_global_scale", "alpha_values", "alpha", "bias"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                setattr(self, name, tensor.to(device, non_blocking=non_blocking))

    def to_cuda(self, non_blocking=False):
        self._move_to(AI_DEVICE, non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        self._move_to("cpu", non_blocking=non_blocking)

    def _reject_adapter(self, weight_dict, kind):
        prefixes = tuple(name.removesuffix(".weight") for name in self.weight_names)
        if any(any(prefix in key for prefix in prefixes) for key in weight_dict):
            raise NotImplementedError(f"NVFP4 fused QKV does not support {kind}")

    def register_lora(self, weight_dict, strength):
        del strength
        self._reject_adapter(weight_dict, "LoRA")

    def update_lora(self, weight_dict, strength):
        self.register_lora(weight_dict, strength)

    def remove_lora(self):
        pass

    def register_diff(self, weight_dict):
        self._reject_adapter(weight_dict, "diff weights")
