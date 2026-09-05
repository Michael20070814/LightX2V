"""FL2VA presentation adapted from H3-World (DiffSynth-Studio)."""

import torch

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"

PRESENTATION_TEXT_TAG = 1
PRESENTATION_VIDEO_TAG = 0


def image_token_counts(processor, images):
    vision = processor.image_processor(images=images, return_tensors="pt")
    merge = int(processor.image_processor.merge_size) ** 2
    counts = [int(vision["image_grid_thw"][i].prod().item()) // merge for i in range(len(images))]
    return vision["pixel_values"], vision["image_grid_thw"], counts


def _text_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _vision_block_ids(tokenizer, pad_token: str, count: int) -> list[int]:
    return [tokenizer.convert_tokens_to_ids(VISION_START)] + [tokenizer.convert_tokens_to_ids(pad_token)] * int(count) + [tokenizer.convert_tokens_to_ids(VISION_END)]


class _Presentation:
    def __init__(self):
        self.ids: list[int] = []
        self.tags: list[int] = []

    def text(self, token_ids: list[int]):
        self.ids += token_ids
        self.tags += [PRESENTATION_TEXT_TAG] * len(token_ids)

    def vision(self, token_ids: list[int]):
        self.ids += token_ids
        self.tags += [PRESENTATION_VIDEO_TAG] * len(token_ids)

    def build(self):
        return (torch.tensor(self.ids, dtype=torch.long), torch.tensor(self.tags, dtype=torch.long))


def presentation_t2va(tokenizer, prompt: str):
    presentation = _Presentation()
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def presentation_fl2va(tokenizer, prompt: str, image_token_counts):
    presentation = _Presentation()
    for index, count in enumerate(image_token_counts, start=1):
        presentation.text(_text_ids(tokenizer, f"<Picture {index}>: "))
        presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()
