"""Rule table adapted from H3-World's code/abot/action_script.py.

Hand-written rules, not a VLM: deterministic, reproducible, zero cost, and
can be generated in real time at inference (a hard requirement for the
streaming case).

The template is a fixed sentence shape with two slots:

    the man <how he moves>, camera <how the view moves>

**The camera clause is generated from COLMAP's measured d_yaw / d_pitch, not
from the IJKL keys.** This is the one place in the whole scheme that can
genuinely add information: if all three clauses were derived from the same
8 bits, this would just be a one-hot encoding spelled out in English (which
is what an Action-Index baseline does).

Thresholds come from measured percentiles over the first 600 clips of the
7872-clip training set (after pooling to the latent grid):

    d_yaw   |v| p50=0.145  p70=0.526  p85=0.945  p95=1.076  (ROT_CLIP caps at 3.0)
    d_pitch |v| p50=0.006  p70=0.020  p85=0.229  p95=0.710

Sign convention verified empirically: J = turn left -> yaw positive,
L = turn right -> yaw negative.

"""

from __future__ import annotations

import numpy as np

from . import abot_action as A

# Global subject anchor, so "the man" in the per-frame clauses has an
# antecedent. What he actually looks like is carried by the <Picture 1>
# visual token in the first frame; this string only needs to give the text
# a stable referent.
SUBJECT_ANCHOR = "A third-person view of a man."

# Motion clauses. Concatenated in a fixed W/S/A/D order so the same key
# combination always produces the same string -- required for per-sentence
# encoding plus a dedup dictionary to work.
MOTION = {
    "W": "walks forward",
    "S": "walks backward",
    "A": "strafes left",
    "D": "strafes right",
}
MOTION_ORDER = ("W", "S", "A", "D")
MOTION_IDLE = "stands still"

# Camera clause thresholds. Three bands instead of a continuous value,
# because language can't express a continuous quantity -- "pans left by
# 0.37" would just add vocabulary noise.
# These thresholds are defined on the **per-frame rate**, not the raw
# summed bin value. bin_to_latent sums the camera channels per bin, and
# _FRAME_PER_TOKEN=(1,4,4,4,4) has unequal bin widths: every 5th step spans
# only 1 frame, so its sum is naturally 1/4 of its neighbors'. Thresholding
# the raw sum would misread 8/37 of the steps as "the camera slowed down"
# when it's really just a narrower bin. Dividing by the bin's frame count
# first fixes that. The numbers below reuse the original calibration on
# 4-frame bins (0.10 / 0.90 / 0.20 / 0.30) divided by 4.
YAW_MIN, YAW_SHARP = 0.025, 0.225
PITCH_MIN = 0.05
TRANS_MIN = 0.075

# ---- 9-bit action representation -------------------------------------------
# The raw recording only has 8 keys; the 9th bit F (fast) is synthesized
# from the pan rate COLMAP measures. Reason: direction can be read off IJKL
# (0.85-0.97 hit rate), but **speed cannot** -- steps with J held split
# 66% slowly / 34% sharply, close to a coin flip; hold duration doesn't give
# a clean band either (the slower first step is just a bin-boundary effect,
# then it plateaus); ICC~=0.475 says it isn't a per-clip constant either.
# Dropping the speed band would make the camera clause a deterministic
# function of IJKL alone, wasting the 6 COLMAP channels we measured, and the
# scheme would degrade back into "one-hot spelled out in English". Adding
# this one bit raises majority-vote lookup accuracy from 0.700 to 0.871.
FAST_COL = "F"
KEYS9 = A.ACTIVE_KEY_COLS + [FAST_COL]
PAN_KEY = {"J": "left", "L": "right"}
TILT_KEY = {"I": "tilts down", "K": "tilts up"}  # measured: K's d_pitch is 95% positive, I's is 95% negative
CAMERA_IDLE = "holds steady"
CAMERA_FOLLOW = "follows him"

# Translation words. Only used when rotation is near zero and the character
# has no movement key held -- in that case the camera's own translation is
# new information independent of the keys. When the character is moving,
# camera translation is just following him, and "follows him" says that
# without restating WASD in different words, which would only bloat the
# annotation and the vocabulary.
TRANS_WORD = {
    "d_x_right": ("drifts right", "drifts left"),
    "d_z_fwd": ("drifts forward", "drifts back"),
    "d_y_down": ("drifts down", "drifts up"),
}


def _camera_rate(pooled: np.ndarray, min_frames: int = 4) -> np.ndarray:
    """Convert the per-bin-summed camera channels into a per-frame rate,
    borrowing from a neighbor to fill out narrow bins.

    Both corrections come from the unequal bin widths of
    _FRAME_PER_TOKEN=(1,4,4,4,4):
      1. Scale -- a 1-frame bin's sum is naturally 1/4 of a 4-frame bin's,
         so thresholding it directly would misread it as "slowed down".
      2. Variance -- a single frame only covers 41 ms, so its estimate is
         noisier and more likely to fall below the threshold by chance,
         producing A-B-A flicker.
    Dividing by the bin's frame count fixes (1); merging bins narrower than
    min_frames with the next bin's estimate fixes (2). The merge only
    affects the rate used for **thresholding** here, not `pooled` itself.
    """
    spans = np.array([e - s for s, e in A.frame_spans(len(pooled))], dtype=np.float32)
    raw = pooled[:, A.NUM_KEYS :]
    rate = raw / spans[:, None]
    for k in range(len(pooled)):
        if spans[k] >= min_frames:
            continue
        j = k + 1 if k + 1 < len(pooled) else k - 1
        rate[k] = (raw[k] + raw[j]) / (spans[k] + spans[j])
    return rate


def keys9(pooled: np.ndarray) -> np.ndarray:
    """[latent_t, 17] -> a [latent_t, 9] 0/1 action tensor.

    The first 8 bits are the raw keys (amax-pooled). The 9th bit, F, is
    derived from the measured pan rate: set to 1 when a pan key is held and
    the per-frame |d_yaw| >= YAW_SHARP. At inference this bit is set
    directly by the user.
    """
    keys = (pooled[:, A.ACTIVE_KEY_INDICES] > 0).astype(np.float32)
    rate = _camera_rate(pooled)
    ji, li = A.ACTIVE_KEY_COLS.index("J"), A.ACTIVE_KEY_COLS.index("L")
    panning = (keys[:, ji] > 0) | (keys[:, li] > 0)
    fast = panning & (np.abs(rate[:, 1]) >= YAW_SHARP)
    return np.concatenate([keys, fast.astype(np.float32)[:, None]], axis=1)


def _purify(on: dict[str, bool], pairs) -> None:
    """Cancel a pair of mutually exclusive keys that ended up both set.

    bin_to_latent takes amax over keys, so a 4-frame window with W pressed
    then S pressed sets both bits to 1. Concatenating them naively would
    produce a self-contradictory annotation like "walks forward and walks
    backward".
    """
    for a, b in pairs:
        if on[a] and on[b]:
            on[a] = on[b] = False


def _motion_clause(on: dict[str, bool]) -> str:
    _purify(on, (("W", "S"), ("A", "D")))
    words = [MOTION[name] for name in MOTION_ORDER if on[name]]
    return " and ".join(words) if words else MOTION_IDLE


def _camera_clause(on: dict[str, bool], moving: bool) -> str:
    _purify(on, (("J", "L"), ("I", "K")))
    parts = []
    for key, side in PAN_KEY.items():
        if on[key]:
            parts.append(f"pans {side} {'sharply' if on[FAST_COL] else 'slowly'}")
    for key, word in TILT_KEY.items():
        if on[key]:
            parts.append(word)
    if parts:
        return " and ".join(parts)
    return CAMERA_FOLLOW if moving else CAMERA_IDLE


def annotate_from_keys9(k9: np.ndarray) -> list[str]:
    """**The annotation is a pure function of these 9 bits** -- training and
    inference therefore run through exactly the same code path.

    The cost is dropping the "drifts" case (the character isn't moving but
    the camera itself is translating, 1.7% of steps): it can't be derived
    from the keys by definition, and inference has no way to produce it
    either, so it folds into "holds steady".
    """
    if k9.ndim != 2 or k9.shape[1] != len(KEYS9):
        raise ValueError(f"expected [latent_t, {len(KEYS9)}], got {list(k9.shape)}")
    out = []
    for row in k9:
        on = {c: bool(row[i] > 0) for i, c in enumerate(KEYS9)}
        motion = _motion_clause(dict(on))
        camera = _camera_clause(dict(on), motion != MOTION_IDLE)
        out.append(f"the man {motion}, camera {camera}")
    return out


def annotate(pooled: np.ndarray, **_ignored) -> list[str]:
    """[latent_t, 17] -> list of annotation strings. Derives the 9-bit
    action first, then runs the pure function above."""
    return annotate_from_keys9(keys9(pooled))
