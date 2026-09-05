"""First-frame inference requests using the training action vocabulary."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lightx2v_train.data.cache_dataset import _single_sample_collate
from lightx2v_train.data.utils import resize_center_crop_frame
from lightx2v_train.data.video_dataset import _build_dataloader
from lightx2v_train.model_zoo.native.minimax_h3 import video_latent_num_frames
from lightx2v_train.utils.registry import DATA_REGISTER

from .action_script import KEYS9, annotate_from_keys9

ACTION_PRESETS = {
    "still": (),
    "forward": ("W",),
    "back": ("S",),
    "strafe-left": ("A",),
    "strafe-right": ("D",),
    "tilt-down": ("I",),
    "tilt-up": ("K",),
    "pan-left": ("J",),
    "pan-right": ("L",),
    "pan-left-fast": ("J", "F"),
    "pan-right-fast": ("L", "F"),
}


def action_script_for_request(request):
    if request["num_frames"] < 5:
        raise ValueError("World inference requires num_frames >= 5.")
    latent_t = video_latent_num_frames(request["num_frames"])
    fields = [key for key in ("action_preset", "action_keys", "action_keys_path") if request.get(key) is not None]
    if len(fields) != 1:
        raise ValueError("Specify exactly one of action_preset, action_keys, action_keys_path.")
    if fields[0] == "action_preset":
        preset = request[fields[0]]
        if preset not in ACTION_PRESETS:
            raise ValueError(f"Unknown action_preset {preset!r}; expected one of {sorted(ACTION_PRESETS)}.")
        rows = [ACTION_PRESETS[preset]] * latent_t
    else:
        rows = request[fields[0]]
        if fields[0] == "action_keys_path":
            rows = json.loads(Path(rows).expanduser().read_text())
    if not isinstance(rows, (list, tuple)) or len(rows) != latent_t:
        raise ValueError(f"Expected exactly {latent_t} action rows for {request['num_frames']} frames.")
    keys = np.zeros((latent_t, len(KEYS9)), dtype=np.int64)
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or any(key not in KEYS9 for key in row):
            raise ValueError(f"Action row {index} must be a list of key names from {KEYS9}.")
        for key in row:
            keys[index, KEYS9.index(key)] = 1
    return annotate_from_keys9(keys)


class WorldInferenceDataset(torch.utils.data.Dataset):
    collate_fn = staticmethod(_single_sample_collate)

    def __init__(self, config):
        samples = config.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("World inference requires a nonempty data.val.samples list.")
        defaults = {key: config[key] for key in ("height", "width", "num_frames") if key in config}
        self.samples = []
        for sample in samples:
            request = {"height": 480, "width": 832, "num_frames": 124, **defaults, **sample}
            for key in ("height", "width", "num_frames"):
                if isinstance(request[key], bool) or int(request[key]) != request[key]:
                    raise ValueError(f"{key} must be an integer.")
                request[key] = int(request[key])
            if any(request[key] <= 0 or request[key] % 32 for key in ("height", "width")):
                raise ValueError("World inference height and width must be positive multiples of 32.")
            if not isinstance(request.get("prompt"), str) or not request["prompt"].strip():
                raise ValueError("Each inference sample requires a nonempty scene prompt.")
            request["first_frame"] = str(Path(request["first_frame"]).expanduser())
            if not Path(request["first_frame"]).is_file():
                raise FileNotFoundError(request["first_frame"])
            request["action_script"] = action_script_for_request(request)
            self.samples.append(request)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        request = self.samples[index]
        with Image.open(request["first_frame"]) as source:
            image = resize_center_crop_frame(source.convert("RGB"), request["height"], request["width"])
        # Preserve the same normalization round trip used by the training processor.
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(1)
        return {**request, "first_frame_tensor": ((pixels + 1) * 0.5).clamp_(0, 1)}


@DATA_REGISTER("minimax_h3_world_infer")
def build_world_inference_data(data_config, train_or_val="val"):
    if train_or_val != "val":
        raise ValueError("minimax_h3_world_infer is an inference dataset.")
    return _build_dataloader(WorldInferenceDataset(data_config), data_config, train_or_val)
