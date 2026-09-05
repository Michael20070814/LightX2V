# MiniMax-H3 World-Model Cache

This branch builds positive-only inputs for future Flow Matching SFT LoRA
training. It does not implement the world-model training consumer. The existing
T2AV training adapter cannot consume FL2VA action conditions; do not select it
to train these caches. No external H3-World or DiffSynth checkout is needed.

ABot-specific dataset, action, encoding and packing code lives under
`lightx2v_train/model_zoo/minimax_h3/`. The common `data/` package retains the
shared video decoding, dataloader and cache-reading infrastructure.

## Environment and Execution

Use a Python environment with the LightX2V training dependencies, Transformers
Qwen3-VL support, and a Diffusers build exposing `AutoencoderKLMiniMaxH3` and
`AutoencoderKLMiniMaxH3Audio`. The video VAE must support single-frame encoding.
ABot reading also uses NumPy, Pillow, imageio and imageio-ffmpeg. The model root
must contain converted Diffusers `vae/` and `audio_vae/` components plus the
Qwen3-VL `text_encoder/`, `tokenizer/`, and `processor/` components. The full
64-layer text checkpoint is read at hidden state 50, before final normalization,
matching the reference retained-50-layer encoder. Original FL2VA custom-code VAE
checkpoints cannot be substituted for the converted VAE directories.

Set `ABOT_DATA_ROOT` to the downloaded dataset root and `H3_MODEL_ROOT` to the
baseline model root using your environment. From `lightx2v_train/`:

```bash
python cache_data.py --config configs/cache/minimax_h3_world.yaml \
    --output_dir "$H3_CACHE_OUTPUT" --save_dtype bf16
```

For the standard two-GPU environment:

```bash
torchrun --standalone --nproc_per_node=2 cache_data.py \
    --config configs/cache/minimax_h3_world.yaml \
    --output_dir "$H3_CACHE_OUTPUT" --save_dtype bf16
```

Every rank scans the same complete sample selection for the shared padding
budget. Data parallelism partitions encoding. No transformer weights are loaded.
The H100 example keeps the text encoder on GPU and offloads VAE components.
Text encoder CPU offload is configurable, but requires sufficient container RAM
for the approximately 63GB weight set; a 32GB container repeatedly evicts and
reloads its mapped weight pages. Check the container limit, not only host RAM.

## Sampling

`data_root/data/<prefix>/<episode>/` contains `video.mp4` and `annotations.tar`.
The archive supplies action.json, caption.json, and COLMAP camera poses. Missing
video, action, or camera annotations are skipped with a reason; missing audio
becomes stereo silence. Audio present in the episode is decoded at the selected
window start, resampled to 32 kHz, and trimmed or padded to the audio latent length.

Episode IDs are sorted then shuffled using `window_seed` (reference default
20260817). Windows are enumerated in rounds across episodes, following the
reference clip plan. `num_clips` is the number of candidate windows, before
incomplete samples are skipped. Increase it to select more candidates. By
default the example selects 64 candidates. Selection is reproducible for an
unchanged inventory. Set `ABOT_NUM_CLIPS=1` for a one-window smoke test.
Downloading additional episodes can change the inventory and its order, so use
a fresh output directory or rebuild.

Source data must be 30fps. The reference 4-of-5 source-frame selection produces
24fps output. Video and action use exactly the same source-frame indices.
Deterministic seeded window starts retain the reference's six-frame reservation,
without writing padded intermediate videos. The reference's jitter and modulo
slot selection are preserved; this is not a guarantee of non-overlapping windows
when requesting many windows from the same episode.

Each configured clip has exactly `17*n+5` frames (default 124, yielding 37 latent
frames). Height and width are configurable multiples of 32 (default 480x832).
All selected frames use LightX2V's aspect-preserving resize and center crop.
The first processed frame supplies both the visual text head and VAE anchor.
There is no random training-time crop or variable-length fallback.

## Encoding and Cache Contract

The first-frame visual tokens and scene prompt use `presentation_fl2va`.
Scene selection prefers `scene_static` with at least 30 words, otherwise the
reference narrative fallback. Action processing retains the reference pose
deltas, episode translation normalization, nonuniform `(1,4,4,4,4)` pooling,
key cancellation, and camera-speed-derived ninth bit. The action sentences are
encoded independently, deduplicated within each encoding process, and appended
in latent-frame order. No `action_cond` feature tensor or negative branch is used.

The usual `cache_data.jsonl`, `cache_meta.json`, and `cache/*.pt` are written.
`cache_skipped.jsonl` records incomplete candidate windows. Tensor files contain
only tensors and basic Python containers and support `weights_only=True`.

| Field | Content |
| --- | --- |
| `inputs.video_latents` | Normalized clean video target `[1,24,T,H/16,W/16]` |
| `inputs.audio_latents` | Normalized stereo audio target `[2,32,A]` |
| `conditioning.positive.prompt_embeds` | FL2VA head followed by action rows `[L,D]` |
| `conditioning.positive.text_token_tags` | Per-row modality tags, preserving vision tags in the head |
| `conditioning.positive.keyframe_cond_anchor` | Patchified first-frame VAE condition with reference noise augmentation |
| `conditioning.positive.action_text_spans` | One half-open text-row interval per video latent frame |
| `conditioning.positive.packed` | Modality positions, float64 position IDs, token tags, sequence boundaries and action binding |
| `meta` | Sample ID, source frame IDs/fingerprint, geometry, schema, cache fingerprint and seed |

The packed action binding includes `action_text_rows`, `action_text_spans_local`,
`action_video_start`, `action_frame_rows`, and `action_real_used`. Position IDs
preserve the reference mirrored temporal offset. `refiner_cu_seqlens` separates
the head and each action sentence. A future consumer must apply the reference
DiT visibility constraints, exclude padding and condition tokens from target
loss, and respect these refiner segments. Saving this metadata does not itself
execute an attention mask.

`action_pad_used_to` is a common total packed-sequence budget, not an action
count. It is scanned from the selected dataset and rounded up to a multiple of
64. Image token counts come from the actual processor, not an assumed formula.
An explicit configured budget must fit every sample. Larger budgets increase
padding work but can accommodate longer descriptions without changing shape.

Reuse checks include source file sizes/mtime, selected frames, action text,
geometry, encoder component file sizes/mtime, and encoding configuration.
These are change detectors, not cryptographic content verification of weights.
Changes require `--overwrite` or a new output directory. Old DiffSynth cache
conversion and partial text-only updates are intentionally outside this tool.

## CPU Checks

From `lightx2v_train/`:

```bash
python -m unittest discover -s tests -p 'test_minimax_h3_world_cache.py' -v
```

The ten CPU tests cover sampling, first-frame consistency, audio handling,
independent action encoding, temporal packing, positive-only cache round trips,
reuse validation and VAE tiling configuration. They use lightweight encoder
substitutes. Action pooling/text, deterministic window starts, and packed tensors
were also compared directly with the reference implementation and matched.

## GPU Smoke Result

On 2026-09-05, one real ABot window was encoded and loaded back through
`CacheDataset` on one H100 80GB. The example configuration was used with
`ABOT_NUM_CLIPS=1` and zero dataloader workers for this smoke run.

- Video: 124 frames, 480x832; cached target `[1,24,37,30,52]`.
- Audio target: `[2,32,207]`.
- FL2VA head plus action embeddings: `[911,5120]` for this sample.
- First-frame anchor: `[390,96]`; action bindings: 37; packed length: 16192.
- Target latents, prompt embeddings and anchor passed finite-value checks.
- Peak allocated GPU memory: 67.795 GiB.
- Total wall time: 271.52 seconds, including 229.3 seconds of initial model
  loading. This is a cold-load single-sample check, not a throughput benchmark.

The run validates encoding, serialization and reading for one sample. Multi-GPU
execution, full-dataset processing, and the future SFT consumer were not tested.

## Source Attribution

`abot_action.py` and `action_script.py` adapt H3-World's `code/abot` action
processing. `world_presentation.py` and `world_packing.py` adapt the FL2VA-only
parts of DiffSynth-Studio's `minimax_h3_text_encoder.py` and
`minimax_h3_audio_video.py`. Dataset window and prompt selection follow
`build_abot_clips.py`; head/action concatenation follows `inject_abot_text.py`.
External path setup, fixed filesystem locations, legacy injection, reference
conditions, inference loops, and training logic were not carried over.
