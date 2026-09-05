"""FL2VA action-time packing adapted from H3-World (DiffSynth-Studio)."""

import numpy as np
import torch


class WorldPackedSequenceBuilder:
    _INTERP = 32
    _T_GROUP = 5
    _FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    _FRAME_RESCALE = 5.0 / 3.0
    _SEQ_ALIGN = 64

    def _aligned_seq_len(self, used: int) -> int:
        return ((used + self._SEQ_ALIGN - 1) // self._SEQ_ALIGN) * self._SEQ_ALIGN

    def _axis_from_sqrt_area(self, dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 0.5
        right = left + ratio
        grid = np.linspace(left, right, dim // patch, endpoint=False) * self._INTERP
        return torch.from_numpy(grid).to(torch.float64)

    def _video_t_grid(self, n: int, origin: float) -> torch.Tensor:
        spans = torch.tensor([self._FRAME_RESCALE * self._FRAME_PER_TOKEN[k % self._T_GROUP] for k in range(n)], dtype=torch.float64)
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    def _temporal_position_span(self, temporal_length: int) -> float:
        spans = np.ones(int(temporal_length), dtype=np.float64) * self._FRAME_RESCALE
        for token_index in range(self._T_GROUP):
            spans[token_index :: self._T_GROUP] *= self._FRAME_PER_TOKEN[token_index]
        return float(spans.sum())

    def _video_t_span(self, n: int) -> float:
        return sum(self._FRAME_RESCALE * self._FRAME_PER_TOKEN[k % self._T_GROUP] for k in range(n))

    def _frame_grid(self, latent_h: int, latent_w: int, sqrt_area):
        h_grid = self._axis_from_sqrt_area(latent_h, 2, sqrt_area)
        w_grid = self._axis_from_sqrt_area(latent_w, 2, sqrt_area)
        hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
        return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_grid

    def _video_grid(self, latent_t: int, frame: torch.Tensor, origin: float) -> torch.Tensor:
        video_g = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
        video_g[:, :, 0] = self._video_t_grid(latent_t, origin)[:, None]
        video_g[:, :, 1:] = frame[None]
        return video_g.reshape(-1, 3)

    def _audio_w_axis(self, w_grid: torch.Tensor, audio_t: int, audio_rows: int) -> torch.Tensor:
        return torch.cat([torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64), torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64)])

    def _build_packed_fl2va(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframe_indices, audio_channel=2, action_text_spans=None, pad_used_to=None):
        """fl2va layout: [text | cond | audio | video | pad].

        pad_used_to: pad the valid length to this fixed value.
        flex_attention is torch.compile'd, and every distinct sequence
        length triggers a recompile; with a different text_len per sample,
        this hits Dynamo's recompile limit within a few steps. Padding to a
        value shared across the whole dataset means it only compiles once.
        The extra rows are zero vectors, masked out by the DiT's attention
        mask (see n_real in _build_action_block_masks).

        action_text_spans: row spans [(start, stop), ...] of the per-latent
        text condition (offsets within the text segment). When given, these
        rows are laid out with a **mirrored offset**: row k is placed at
        t = (text_len - video span) + s_k, so its t-offset from frame k's t
        is a constant negative number. This makes the binding signal a
        single uniform rule ("look back a fixed distance"), without
        breaking the pretraining invariant that text always comes before
        video in t -- placing it directly at the frame's own t would make
        the text<->video t-offset both zero and positive, a configuration
        pretraining never saw.
        """
        frame_rows = (latent_h // 2) * (latent_w // 2)
        video_rows = latent_t * frame_rows
        audio_rows = audio_t * audio_channel
        num_keyframes = len(keyframe_indices)
        cond_rows = num_keyframes * frame_rows
        used = text_len + cond_rows + audio_rows + video_rows
        real_used = used
        if pad_used_to is not None:
            if pad_used_to < used:
                raise ValueError(f"pad_used_to {pad_used_to} is smaller than the actual length {used}")
            used = int(pad_used_to)
        seq_len = self._aligned_seq_len(used)

        text_sl = slice(0, text_len)
        cond_sl = slice(text_len, text_len + cond_rows)
        audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
        video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

        # img_pos covers both cond AND video rows, conditions first
        img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), torch.arange(video_sl.start, video_sl.stop)])
        audio_pos = torch.arange(audio_sl.start, audio_sl.stop)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

        action_text_rows = None
        if action_text_spans is not None:
            if len(action_text_spans) != latent_t:
                raise ValueError(f"action_text_spans should have {latent_t} entries, got {len(action_text_spans)}")
            prev = 0
            for k, (lo, hi) in enumerate(action_text_spans):
                if not (0 <= lo < hi <= text_len):
                    raise ValueError(f"span {k} ({lo},{hi}) is outside the text segment [0,{text_len})")
                if lo < prev:
                    raise ValueError(f"span {k} ({lo},{hi}) overlaps or is out of order with the previous one")
                prev = hi
            s_k = self._video_t_grid(latent_t, 0.0)
            origin = float(text_len) - float(s_k[-1]) - 1.0  # back off by one step so the last row doesn't land on the same t as the first-frame cond
            head_end = int(action_text_spans[0][0])
            if origin < head_end:
                raise ValueError(
                    f"the annotation block would need t in [{origin:.0f}, {origin + float(s_k[-1]):.0f}], "
                    f"which overlaps the preceding image/anchor segment [0,{head_end}); "
                    f"lengthen the text segment or shorten the annotations"
                )
            rows = []
            for k, (lo, hi) in enumerate(action_text_spans):
                g[lo:hi, 0] = origin + float(s_k[k])
                rows.append((lo, hi))
            action_text_rows = torch.tensor(rows, dtype=torch.long)

        sqrt_area = np.sqrt(latent_h * latent_w)
        frame, w_grid = self._frame_grid(latent_h, latent_w, sqrt_area)

        # Condition rows: temporal position depends on frame_index
        temporal_span = self._temporal_position_span(latent_t)
        for i, idx in enumerate(keyframe_indices):
            sl = slice(i * frame_rows, (i + 1) * frame_rows)
            if idx == 0:
                cond_t = float(text_len)
            else:  # idx == -1
                cond_t = float(text_len) + temporal_span - self._FRAME_RESCALE
            cond_g = torch.empty(frame_rows, 3, dtype=torch.float64)
            cond_g[:, 0] = cond_t
            cond_g[:, 1:] = frame
            g[cond_sl.start + sl.start : cond_sl.start + sl.stop] = cond_g

        g[video_sl] = self._video_grid(latent_t, frame, float(text_len))
        g[audio_sl, 0] = (float(text_len) + torch.arange(audio_t, dtype=torch.float64)).repeat(audio_channel)
        g[audio_sl, 2] = self._audio_w_axis(w_grid, audio_t, audio_rows)

        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[text_sl] = 1
        token_tags[audio_sl] = 2
        token_tags[img_pos] = 0  # both cond and video rows are tagged as video (0)

        out = {
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": torch.arange(0, text_len),
            "img_position_ids": g[None],
            "token_tags": token_tags,
            "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
            "seq_len": seq_len,
        }
        if action_text_rows is not None:
            out["action_text_rows"] = action_text_rows
            # Local spans within the text segment, used to split the Token Refiner's segments (see the refiner_cu note below).
            out["action_text_spans_local"] = [(int(lo), int(hi)) for lo, hi in action_text_spans]
            out["action_video_start"] = video_sl.start
            out["action_frame_rows"] = frame_rows
            out["action_real_used"] = real_used
        return out
