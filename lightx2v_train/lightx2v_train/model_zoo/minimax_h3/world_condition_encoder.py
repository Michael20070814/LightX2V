"""FL2VA visual head and independent action sentences, without DiffSynth."""

import torch

from .condition_encoder import MiniMaxH3ConditionEncoder
from .world_presentation import image_token_counts, presentation_fl2va, presentation_t2va


def concatenate_actions(head, head_tags, script, embeddings):
    parts, spans = [head], []
    cursor = head.shape[0]
    for sentence in script:
        rows = embeddings[sentence].to(device=head.device, dtype=head.dtype)
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != head.shape[1]:
            raise ValueError("Action embeddings must be nonempty [tokens, hidden_size] tensors.")
        spans.append((cursor, cursor + rows.shape[0]))
        cursor += rows.shape[0]
        parts.append(rows)
    tags = torch.cat([head_tags, torch.ones(cursor - head.shape[0], dtype=torch.long, device=head_tags.device)])
    return torch.cat(parts), tags, spans


class MiniMaxH3WorldConditionEncoder(MiniMaxH3ConditionEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.text_encoder_layer >= self.encoder.config.text_config.num_hidden_layers:
            raise ValueError("FL2VA requires an intermediate, pre-final-norm hidden state (normally layer 50 of 64).")
        self.action_embeddings = {}

    @torch.inference_mode()
    def _encode_ids(self, ids, pixel_values=None, grid=None):
        ids = ids.unsqueeze(0).to(self.device)
        self._activate()
        try:
            mm_ids = torch.zeros_like(ids, dtype=torch.int32)
            mm_ids[ids == self.encoder.config.image_token_id] = 1
            output = self.encoder.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                mm_token_type_ids=mm_ids,
                pixel_values=None if pixel_values is None else pixel_values.to(self.device, self.dtype),
                image_grid_thw=None if grid is None else grid.to(self.device, torch.long),
                use_cache=False,
                output_hidden_states=True,
            )
            # Layer 50 before the next layer's norm equals the retained-50-layer
            # H3-World encoder with its final norm replaced by Identity.
            return output.hidden_states[self.text_encoder_layer][0].to("cpu", self.dtype).contiguous()
        finally:
            self._offload()

    def encode_world(self, prompt, image, script):
        pixels, grid, counts = image_token_counts(self.processor, [image])
        ids, tags = presentation_fl2va(self.tokenizer, prompt, counts)
        head = self._encode_ids(ids, pixels, grid)
        for sentence in dict.fromkeys(script):
            if sentence not in self.action_embeddings:
                action_ids, _ = presentation_t2va(self.tokenizer, sentence)
                self.action_embeddings[sentence] = self._encode_ids(action_ids)
        embeddings, tags, spans = concatenate_actions(head, tags, script, self.action_embeddings)
        return {"prompt_embeds": embeddings, "text_token_tags": tags, "action_text_spans": spans}
