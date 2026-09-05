# MiniMax-H3 World-Model SFT

LightX2V provides two distinct H3 training paths:

| Model | Conditions | Objective | Example |
| --- | --- | --- | --- |
| `minimax_h3_t2av` | Scene text | Existing continuous-sigma joint flow, unweighted modality MSE | `configs/train/flow/minimax_h3_t2av_lora.yaml` |
| `minimax_h3_world` | FL2VA first frame, scene, independently encoded actions bound to latent time | Discrete 1000-step joint flow with timestep-weighted modality MSE | `configs/train/flow/minimax_h3_world_lora_tp.yaml` |

The world path consumes LightX2V **training data caches**, not H3-World legacy
cache files. Build them using [the world cache workflow](minimax_h3_world_cache.md).
Training only loads the DiT. It does not load the VAE or text encoder and does
not import or require the external H3-World repository. The supported training
topology is two H100 80GB GPUs, TP=2, with one shared sample per step. DDP, FSDP
and sequence parallelism are not combined with TP in this implementation.

## Run

Activate an environment with the project training dependencies, the Diffusers
`MiniMaxH3Transformer3DModel`, PEFT, and CUDA PyTorch with compiled FlexAttention
(validated environment: PyTorch 2.8.0+cu128). Supply these environment variables:

```bash
export H3_MODEL_ROOT=/your/converted/MiniMax-H3
export H3_CACHE_MANIFEST=/your/world-cache/cache_data.jsonl
export H3_TRAIN_OUTPUT=/your/output/world-sft
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/run_minimax_h3_world_lora_tp.sh
```

Run from `lightx2v_train/`. Set `PYTHON` to an interpreter path when needed.
The model root must contain the converted Diffusers `transformer/config.json`
and safetensors weights, rather than the original custom FL2VA transformer.
All resource and output paths are supplied by configuration or environment.
The existing cache fixes clip geometry and the shared padding budget.

The example defaults to 3000 optimizer steps, rank=32, alpha=32, learning
rate=1e-4, gradient checkpointing, and a checkpoint every 100 steps.
`H3_TRAIN_ITERS`, `H3_SAVE_EVERY`, `H3_LORA_RANK`, and `H3_LORA_ALPHA` override
those defaults. Other settings are editable in the YAML. Both ranks use the
same deterministic dataset ordering, sampled timestep index, and noise.

## Objective and Conditioning

Uniformly sample an index from the 1000-element table with base sigma
`linspace(1, 0, 1001)[:-1]`. Independently configured video and audio shifts
transform this table with `shift*sigma / (1+(shift-1)*sigma)`. Both shifts
default to **2.22**, matching the referenced H3-World training entrypoint;
values must exceed one so sampling favors higher noise. This differs from
the existing T2AV defaults and does not change them.

For each modality, compute `y = exp(-2*(sigma-0.5)^2)` over its entire table,
then normalize `y-y.min()` to mean one. Multiply the modality MSE by the
selected table weight. The clean-noise sign of the DiT output is reconciled
with the framework's noise-clean target. Shared framework noise mixing,
optimizer, accumulation, logging, and training loop remain in use.

All target video frames, including the first, and stereo audio are noised.
Video and audio loss coefficients each default to one. Missing source audio
is already encoded as silence in the training cache. The separately cached
first-frame anchor retains its cached augmentation and uses the corresponding
conditioning timestep. Neither anchor nor padding rows are prediction targets.

The refiner processes the FL2VA head and each action sentence independently.
In every DiT layer, action A_k can be read only by its own sentence and video
latent V_k. A_k can read the scene, anchor, audio and V_k, but no other action
or video latent. Video-to-video attention remains unrestricted; indirect
action propagation through V_k is intentional. Padding can attend only to
itself. GPU training uses a compiled FlexAttention block mask; dense SDPA is
restricted to small validation sequences.

## LoRA, Tensor Parallelism and Checkpoints

Only backbone attention `qkv_proj` and `o_proj` receive LoRA. QKV is fused,
sharing one low-rank A factor across Q/K/V, as in H3-World. FFNs, refiner,
conditioning projections and every pretrained parameter remain frozen.
Fused QKV output rows use H3-World's `[head, QKV, head_dim]` order. TP
partitions contiguous groups of heads, including the corresponding LoRA B rows.
Adapters record `qkv_layout: head_qkv_dim` in `adapter_config.json`; older
adapters without this field use a different layout and are rejected on resume.
Parameters are loaded directly from local safetensors slices into each GPU's
shard to avoid a complete CPU model allocation.

TP splits attention heads, fused QKV, SwiGLU FFNs and the large AdaLN
projections. Sequence activations are replicated. Autograd collectives and
adapter gradient reductions preserve single-model gradient semantics. Gradient
clipping counts each full logical parameter once, including replicated factors.

Each completed `checkpoint-NNNNNNNNN/` contains:

- `pytorch_lora_weights.safetensors`: gathered, unsharded world LoRA factors.
- `adapter_config.json`: rank, alpha, target modules and world adapter format.
- `training_state_rank0.pt` and `training_state_rank1.pt`: local optimizer,
  LR scheduler, iteration, topology and PyTorch RNG states.
- `config.yaml` and `_SUCCESS`: configuration and completion marker.

Auto-resume selects the latest completed checkpoint and restores TP=2 state.
Keep the same dataset manifest, accumulation factor and optimizer settings
when resuming. Dataset position is recovered from the optimizer iteration.
The exported adapter uses LightX2V world-model module names, including fused
QKV. It can be reloaded by the world training model; it is not a drop-in
adapter for the separate T2AV model or an H3-World inference script. Use the
[world inference entrypoint](minimax_h3_world_inference.md) for action-conditioned
generation with the training-side transformer.

## Verification

```bash
python -m unittest discover -s tests -p 'test_minimax_h3_world*.py' -v
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
    tests/test_minimax_h3_world_training.py
```

CPU checks cover the discrete distribution and weights, visibility for every
pair of token types, malformed cache rejection, refiner isolation, and the
LoRA gradient scope. The two-GPU probe compares outputs, both LoRA factors'
gradients and global clipping with a single-model reference, and checks
FlexAttention against dense attention.

Before the QKV layout change, on 2026-09-05, the complete pretrained DiT trained on two H100 80GB GPUs
using one real ABot cache (124 frames, 480x832, 16192 packed rows). Two optimizer
steps produced finite losses of 0.389412 and 0.229380 and saved both checkpoints.
The exported adapter contains 200 tensors / 63,078,400 parameters, all finite;
all 200 tensors changed during the second update. Restarting from step 1 and
repeating step 2 reproduced all 200 exported tensors exactly (maximum absolute
difference 0). This verifies the training and resume workflow on one sample;
it is not a convergence or dataset-scale quality evaluation.
