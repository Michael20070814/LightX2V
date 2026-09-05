# MiniMax-H3 World LoRA Inference

Run action-conditioned first-frame video/audio generation with the same
`WorldTransformer`, directed action mask and independently segmented token
refiner used by world SFT. No H3-World checkout is needed at runtime.

## Run

Use the training environment with MiniMax-H3 Diffusers modules, Transformers
Qwen3-VL, PEFT, Accelerate, imageio and imageio-ffmpeg. The base model must be
the converted Diffusers root containing `transformer/`, `vae/`, `audio_vae/`,
`text_encoder/`, `tokenizer/` and `processor/`.

From `lightx2v_train/`, configure paths and a scene description:

```bash
export H3_MODEL_ROOT=/your/converted/MiniMax-H3
export H3_LORA_PATH=/your/training/checkpoint-000000100
export H3_FIRST_FRAME=/your/first_frame.png
export H3_SCENE_PROMPT='A stone-paved village street lined with houses.'
export H3_INFER_OUTPUT=/your/world-inference
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/run_minimax_h3_world_infer.sh
```

`PYTHON` selects the interpreter; `H3_INFER_CONFIG` selects a different YAML.
The example uses TP=2 for two H100 80GB GPUs, with CPU offload for encoders
and the DiT between conditioning, denoising and decoding. CPU RAM must hold
the offloaded weights for both processes. This is stage-level DiT offload:
each GPU must still fit its complete DiT shard during sampling.

For single-process inference, set `distributed.tensor_parallel.enabled: false`
and run `python infer.py --config <yaml>`. The device must fit the complete
DiT; a single H100 is not guaranteed to have sufficient memory. Select the
device with `CUDA_VISIBLE_DEVICES`. FSDP2, SP and DP combinations are not
supported on this path.

Output files are `00000.mp4`, `00001.mp4`, etc., in `inference.output_dir`,
in sample order. Each contains 24fps video and generated stereo audio at
32kHz. Audio is padded/trimmed at mux time to match video duration. Existing
files with the same names are replaced after successful encoding.

## Requests and Actions

The example configuration is `configs/infer/minimax_h3_world_lora.yaml`.
Requests are listed in `data.val.samples`. Each requires `first_frame`,
`prompt` (scene description) and exactly one action input:

- `action_preset`: one held key combination for the entire clip.
- `action_keys`: a list of key-name lists, one per video latent frame.
- `action_keys_path`: a configurable JSON file containing that same list.

Presets: `still`, `forward`, `back`, `strafe-left`, `strafe-right`, `tilt-down`,
`tilt-up`, `pan-left`, `pan-right`, `pan-left-fast`, `pan-right-fast`.

Key names are `W`, `A`, `S`, `D`, `I`, `J`, `K`, `L`, `F`; `[]` means still.
Use the training semantics: I/K tilt down/up, J/L pan left/right, and F marks
fast movement. Conflicting keys are resolved by the existing training action
annotator. Each action sentence is encoded independently, including when
the same sentence is reused at multiple latent times.

For example, the minimum five-frame request has two latent steps:

```yaml
data:
    val:
        name: minimax_h3_world_infer
        num_workers: 0
        height: 480
        width: 832
        num_frames: 5
        samples:
            - first_frame: /your/first_frame.png
              prompt: A stone-paved village street lined with houses.
              action_keys: [[W], [L, F]]
              seed: 7
```

Height, width and num_frames may be overridden per sample. Dimensions must
be positive multiples of 32; frames must be `17k+5`, with `k >= 0`. Latent
length is `5k+2`: 124 frames need 37 action rows, 243 need 72, and 481 need
142. Wrong lengths or unknown keys are rejected, never silently repeated or
truncated. The sequence is on the latent timeline, not the raw-frame timeline.

The first frame uses exactly the training bilinear aspect-preserving resize
and center crop. The processed pixels feed both the VAE and the FL2VA visual
text encoder. `model.imgvid_cond_noise_aug` defaults to 0.999. Anchor noise
uses the request seed unless `model.condition_seed` is explicitly set.
Set it to the cache's condition seed when comparing conditioning with a cache.

## Sampling and LoRA

Only positive conditioning is supported. `enable_cfg` must be false and any
guidance scale must be 1. No negative scene or action encoder pass is run.
`num_inference_steps` counts DiT evaluations (default 50); the terminal zero
sigma is additional. Video/audio shifts default to 12/3, matching the
reference inference entrypoint, and are independently configurable through
`video_flow_shift` / `audio_flow_shift`. These inference defaults differ from
the training timestep distribution. Video/audio noise follows the reference's
CPU RNG initialization; denoising accumulates in FP32.

`model.action_pad_used_to` optionally sets a fixed packing budget, such as
the training cache budget. When omitted, packing rounds up to a multiple of
64 per request. The mask is built once per request and reused across steps.

`inference.lora_config.path` accepts either:

- A LightX2V world checkpoint directory, or its `pytorch_lora_weights.safetensors`
  file with adjacent `adapter_config.json`. The metadata must declare
  `qkv_layout: head_qkv_dim`; old pre-layout-change checkpoints are rejected.
- An H3-World text-action `.safetensors` file with backbone
  `blocks.*.attn.qkv_proj/out_proj` A/B factors. Module names are mapped to
  the training transformer. Both now use `[head, QKV, head_dim]`, so no row
  permutation is needed. The default is alpha=rank, matching the reference
  script's unit `B @ A` scale; `lora_config.alpha` may specify another value.

`lora_config.strength` scales the adapter (default 1). LightX2V alpha is read
from its metadata; a conflicting override is rejected. All backbone attention
layers and both A/B factors must be present with valid shapes and finite
values. FiLM checkpoints and additional trainable tensors are not supported.
Remove `lora_config` to run the base weights with the same action conditions.

## Verification

```bash
python -m unittest discover -s tests -p 'test_minimax_h3_world*.py' -v
```

CPU tests cover action lengths, training/inference first-frame equality,
native/reference adapter loading and strength, old-layout rejection, schedule
direction, packing round trips, decode normalization, a real small DiT sampling
loop with simulated encoders, and MP4 muxing. Full-size GPU generation and
output quality have not been validated for this inference implementation.
