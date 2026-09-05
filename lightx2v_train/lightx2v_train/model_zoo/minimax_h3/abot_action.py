"""ABot action processing adapted from H3-World's code/abot/abot_action.py.

The following reference semantics are retained for cache compatibility.

ABot-World-Explorer action labels -> a conditioning tensor on the MiniMax-H3
latent timeline.

Same idea as the V-Rising version of this file, but with three real
differences, all found empirically:

1. **The four `delta_*` vectors in `action.json` are all zero** (100% zero
   across a 296-clip sample). The continuous motion signal has to be
   recovered from COLMAP's `sparse/0/images.txt` instead.
   Pose coverage is 1800/1800 = 100%, one pose per frame.

2. **COLMAP's scale is arbitrary per episode.** Holding only W, the
   per-frame displacement median spans 0.0045 to 0.1488 across 40 sampled
   episodes (33x). So translation must be **normalized per episode**;
   rotation is an angle and has no scale problem.

3. **No `resample_to_video_timeline` needed.** The V-Rising side has an
   81-frame @16fps sidecar against a video resampled to 107 frames @24fps,
   which needs a resampling step. Here we cut both the video and the action
   with the same explicit frame-index list at slicing time, so the two are
   frame-aligned by construction; `LoadVideo` reads it back as the identity
   mapping (verified: zero duplicate frames).

Channel layout (`ACTION_DIM = 17`):

    0..10   11 binary keys, amax within the window
    11..13  rotation deltas pitch/yaw/roll, degrees, summed within the
            window, then divided by ROT_SCALE
    14..16  translation deltas (x right, y down, z forward) in the previous
            frame's camera frame, already normalized per episode, summed
            within the window, then divided by TRA_SCALE

Why amax for binary and sum for continuous: a key press is a pulse; if it
was held anywhere in the window it counts as held, and averaging would
dilute it into a meaningless 0.25-style intensity. A delta is an additive
physical quantity; a latent token spanning 4 frames should get the total
displacement over those 4 frames. Note H3's frame grouping is non-uniform
(1,4,4,4,4): the first token in each group only spans 1 frame, so its sum
is naturally 1/4 of the others'. That's real information (the token really
does represent only 1 frame), not an artifact to smooth away.
"""

import itertools
import json
import tarfile

import numpy as np

# ---- H3 video VAE temporal grouping convention, matches h3_action.py / PackedSequenceBuilder ----
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_T_GROUP = 5

# ---- Dataset-native schema. control_scheme = WASD_QE_locomotion_IJKL_rotation ----
# Q/E/Space were never pressed across a 40-episode sample, but that's only
# 296/30969 clips, not enough to assert they're zero for the full set. And
# keeping 3 always-zero channels only costs 3*5376*50 parameters that never
# get a gradient, in exchange for the schema matching the dataset's native
# definition bit for bit. Better to keep them.
KEY_COLS = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"]
ACTIVE_KEY_COLS = ["W", "A", "S", "D", "I", "J", "K", "L"]
ACTIVE_KEY_INDICES = [KEY_COLS.index(name) for name in ACTIVE_KEY_COLS]
ROT_COLS = ["d_pitch", "d_yaw", "d_roll"]
TRA_COLS = ["d_x_right", "d_y_down", "d_z_fwd"]
ACTION_COLS = KEY_COLS + ROT_COLS + TRA_COLS
NUM_KEYS = len(KEY_COLS)
ACTION_DIM = len(ACTION_COLS)

# Bring the summed magnitudes down to O(1). Measured over 40 episodes,
# binned in windows of 4 frames: rotation p95 median 2.24 deg, max median
# 3.73; normalized translation p95 is about 4 (roughly 1 per frame).
ROT_SCALE = 4.0
TRA_SCALE = 4.0

# 30fps -> 24fps: drop the 5th of every 5 source frames. This rule (rather
# than replicating H3's own round() mapping) is exact under ffmpeg's select
# filter and exactly invertible, so "which source frame does video frame j
# come from" is defined by us, not inferred from H3's internals.
_KEEP_OF, _DROP_EVERY = 4, 5


def window_offsets(n_out: int):
    """Output frame j -> source-frame offset relative to the window start."""
    return [(j // _KEEP_OF) * _DROP_EVERY + (j % _KEEP_OF) for j in range(n_out)]


def window_span(n_out: int) -> int:
    """How many source frames an n_out-frame output window consumes."""
    return window_offsets(n_out)[-1] + 1


def latent_t_for(num_frames: int) -> int:
    return ((num_frames - 5) // 17) * _T_GROUP + 2


def frame_spans(latent_t: int):
    spans = [_FRAME_PER_TOKEN[k % _T_GROUP] for k in range(latent_t)]
    ends = list(itertools.accumulate(spans))
    return list(zip([0] + ends[:-1], ends))


# --------------------------------------------------------------------------- #
# COLMAP
# --------------------------------------------------------------------------- #
def quat_to_R(q: np.ndarray) -> np.ndarray:
    """Hamilton (w,x,y,z) -> 3x3. COLMAP stores the world->camera rotation R_cw."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _parse_images_txt(blob: bytes):
    """images.txt -> {NAME (no extension): (quat[4], t_cw[3])}.

    Only the first line of each record is used; the second line is 2D
    observations, which are unused here since points3D.txt is empty for
    this dataset. IMAGE_ID is not guaranteed contiguous, so NAME is always
    used as the key.
    """
    out = {}
    for line in blob.decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 10:
            continue  # observation line
        try:
            quat = np.array([float(v) for v in p[1:5]])
            tvec = np.array([float(v) for v in p[5:8]])
        except ValueError:
            continue  # guard in case an observation line happens to be long enough
        out[p[-1].rsplit(".", 1)[0]] = (quat, tvec)
    return out


def read_episode(tar_path: str):
    """Read one annotations.tar and return per-source-frame keys and pose.

    Returns a dict:
        keys     [T, 11] float32   0/1
        R_cw     [T, 3, 3]
        C_world  [T, 3]            camera center = -R_cw^T t_cw
        fps, total_frames, control_scheme
    Only frames covered by both action.json and images.txt are kept, ordered
    by action.json's frame_id (verified: NAME lines up with frame_id
    one-to-one).
    """
    with tarfile.open(tar_path) as t:
        members = {m.name: m for m in t.getmembers()}
        allow = {"action.json", "caption.json", "sparse/0/cameras.txt", "sparse/0/images.txt", "sparse/0/points3D.txt"}
        bad = [n for n, m in members.items() if n not in allow or not m.isfile()]
        if bad:
            raise ValueError(f"{tar_path}: unexpected tar members {bad[:3]}")
        act = json.load(t.extractfile("action.json"))
        poses = _parse_images_txt(t.extractfile("sparse/0/images.txt").read())
        cap = json.load(t.extractfile("caption.json"))

    frames = act["frames"]
    order = [f["frame_id"] for f in frames]
    keep = [i for i, n in enumerate(order) if n in poses]
    if len(keep) != len(order):
        # The dataset docs explicitly say not to assume every frame has a
        # pose. In practice we've only ever seen 40/40 full coverage, but if
        # a gap does show up, drop the whole episode rather than silently
        # filling zeros and manufacturing a fake "stands still" action.
        raise ValueError(f"{tar_path}: pose coverage {len(keep)}/{len(order)}, not complete")

    keys = np.zeros((len(frames), NUM_KEYS), dtype=np.float32)
    for i, fr in enumerate(frames):
        k = fr["keys"]
        for c, name in enumerate(KEY_COLS):
            if k.get(name):
                keys[i, c] = 1.0

    R = np.stack([quat_to_R(poses[n][0]) for n in order])
    T = np.stack([poses[n][1] for n in order])
    C = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -T)
    return dict(keys=keys, R_cw=R, C_world=C, fps=float(act.get("fps", 30.0)), total_frames=len(frames), control_scheme=act.get("control_scheme"), caption=cap)


def pose_deltas(R: np.ndarray, C: np.ndarray):
    """Pose sequence -> step-to-step deltas. Row 0 is always 0 (no "previous frame").

    Translation is expressed in **the previous frame's camera frame**
    (+x right / +y down / +z forward, COLMAP convention), so pressing W to
    move forward lands on +z, matching the key semantics directly and
    staying independent of world-frame orientation. Rotation is the
    relative rotation R_rel = R[i] @ R[i-1]^T, decomposed into
    pitch/yaw/roll (degrees).

    The deltas are computed on the sequence **after frame selection**, not
    by computing per-source-frame deltas first and summing; this keeps the
    result exact across dropped source frames instead of relying on a
    small-angle additivity approximation.
    """
    n = len(C)
    dC = np.zeros((n, 3), dtype=np.float64)
    dE = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return dC, dE
    dC[1:] = np.einsum("nij,nj->ni", R[:-1], np.diff(C, axis=0))
    Rrel = np.einsum("nij,njk->nik", R[1:], R[:-1].transpose(0, 2, 1))
    dE[1:, 0] = np.degrees(np.arcsin(np.clip(-Rrel[:, 1, 2], -1.0, 1.0)))  # pitch
    dE[1:, 1] = np.degrees(np.arctan2(Rrel[:, 0, 2], Rrel[:, 2, 2]))  # yaw
    dE[1:, 2] = np.degrees(np.arctan2(Rrel[:, 1, 0], Rrel[:, 1, 1]))  # roll
    return dC, dE


def episode_translation_scale(ep: dict, n_probe_out: int = 1400) -> float:
    """This episode's "typical per-output-frame displacement", used to cancel
    out COLMAP's arbitrary scale.

    Computed on the timeline **after frame selection**, so the unit is "per
    24fps frame", matching training. Uses the median rather than the mean
    so a few reconstruction jumps don't dominate it.
    """
    idx = np.array(window_offsets(n_probe_out))
    idx = idx[idx < ep["total_frames"]]
    dC, _ = pose_deltas(ep["R_cw"][idx], ep["C_world"][idx])
    spd = np.linalg.norm(dC[1:], axis=1)
    med = float(np.median(spd)) if len(spd) else 0.0
    return med


# --------------------------------------------------------------------------- #
# Window slicing -> per-frame action matrix
# --------------------------------------------------------------------------- #
def window_action_matrix(ep: dict, start: int, n_out: int, scale: float) -> np.ndarray:
    """[n_out, 17], frame-aligned with the sliced mp4 (output frame j <->
    source frame start+offsets[j]).

    Translation has already been divided by `scale` (per-episode
    normalization); rotation keeps its raw degrees. Neither has been
    divided by ROT_SCALE / TRA_SCALE yet; that happens at binning time, so
    changing those scales later doesn't require rerunning this step.
    """
    idx = np.array([start + o for o in window_offsets(n_out)])
    if idx[-1] >= ep["total_frames"]:
        raise ValueError(f"window out of range: needs source frame {idx[-1]}, only have {ep['total_frames']}")
    dC, dE = pose_deltas(ep["R_cw"][idx], ep["C_world"][idx])
    if scale > 1e-9:
        dC = dC / scale
    else:
        dC = np.zeros_like(dC)  # the whole clip barely moves; normalizing would be meaningless
    return np.concatenate([ep["keys"][idx], dE, dC], axis=1).astype(np.float32)


# Symmetric clipping thresholds for the continuous channels. **Both are
# necessary**: the pose sequence has reconstruction jumps mixed in, and
# without clipping a small number of samples would feed wildly out-of-scale
# gradients into the zero-initialized embedder.
#
# Measured over 2225 clips (after binning and scaling):
#   translation p99 ~= 2.4, unclipped max = 47.5  -> TRA_CLIP=4.0 only clips 0.174% of elements
#   rotation p99 ~= 1.2, unclipped max = 21.2      -> ROT_CLIP=3.0 only clips 0.020% / 10 clips
#
# The thresholds aren't arbitrary: the rotation quantile curve has a clear
# knee between p99.9=1.70 and p99.99=9.59, which says the extreme values
# are discrete outliers rather than a continuous heavy tail. A stronger
# check: these extremes don't line up with the keys; of the 48 tokens with
# yaw>=3.0, only 15% also have J/L held, and a real fast turn can't happen
# without holding a turn key. So they're reconstruction jumps and should be
# clipped. 3.0 = 12 deg/bin = 72 deg/s, which still fully preserves a real
# fast turn.
TRA_CLIP = 4.0
ROT_CLIP = 3.0


def bin_to_latent(mat: np.ndarray, latent_t: int, rot_scale: float = ROT_SCALE, tra_scale: float = TRA_SCALE, tra_clip: float = TRA_CLIP, rot_clip: float = ROT_CLIP) -> np.ndarray:
    """[num_frames, 17] -> [latent_t, 17], binned by H3's non-uniform grouping.

    Binary segments take amax, continuous segments take sum then scale, and
    both continuous segments get symmetric clipping.
    """
    spans = frame_spans(latent_t)
    if spans[-1][1] != len(mat):
        raise ValueError(f"binned span total {spans[-1][1]} != action frame count {len(mat)}")
    out = np.zeros((latent_t, mat.shape[1]), dtype=np.float32)
    for k, (s, e) in enumerate(spans):
        out[k, :NUM_KEYS] = mat[s:e, :NUM_KEYS].max(axis=0)
        out[k, NUM_KEYS:] = mat[s:e, NUM_KEYS:].sum(axis=0)
    out[:, NUM_KEYS : NUM_KEYS + 3] /= rot_scale
    out[:, NUM_KEYS + 3 :] /= tra_scale
    if rot_clip:
        np.clip(out[:, NUM_KEYS : NUM_KEYS + 3], -rot_clip, rot_clip, out=out[:, NUM_KEYS : NUM_KEYS + 3])
    if tra_clip:
        np.clip(out[:, NUM_KEYS + 3 :], -tra_clip, tra_clip, out=out[:, NUM_KEYS + 3 :])
    return out
