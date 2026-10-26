"""Per-chunk sparse-attention budget for FlashVSR's streaming denoiser.

FlashVSR denoises a video in temporal chunks (the first chunk covers the
first 25 input frames, every later chunk 8 more). For each chunk the sparse
self-attention uses three knobs:

* ``topk_ratio``  -- how many key windows each query window keeps (sparsity)
* ``kv_ratio``    -- how many earlier chunks stay in the KV cache
* ``local_range`` -- the spatial neighbourhood (in 8x8 windows) a window may see

Upstream FlashVSR uses one fixed value for the whole video. ``MotionBudget``
varies them per chunk from a cheap motion estimate of the low-resolution
input. This is FlashVSR-Linux's own heuristic, written from the general idea
"spend attention where the content moves"; it is not the TE-Speed-FlashVSR
algorithm (that one is closed source), and its quality/speed trade-off has to
be judged on real videos. ``intensity = 0`` reproduces upstream behaviour.

Rules (s = normalised motion of the chunk, 0 = static, 1 = strong motion):

* topk:  scaled by ``1 + 0.25 * intensity * (2s - 1)``  (at most +-25 %)
  static chunks keep fewer windows (redundant content), moving chunks more.
* local: 11 for chunks with s >= 0.5, 9 otherwise (upstream: 9 = sharper,
  11 = more stable), only when intensity > 0 and the user did not pin it.
* kv:    reduced by ``round(intensity * s)`` chunks (never below 2): old frames
  are less useful when the scene moves a lot.
"""

import torch
import torch.nn.functional as F

PROFILES = {
    # profile: (sparse_ratio, kv_ratio, local_range)
    "detail": (2.0, 3.0, 9),
    "balanced": (2.0, 3.0, 11),
    "throughput": (1.5, 3.0, 11),
}

MOTION_LOW = 0.004   # mean |frame difference| (0..1 RGB) treated as static
MOTION_HIGH = 0.04   # ... treated as strong motion


@torch.no_grad()
def frame_motion(frames, max_side=96):
    """Mean absolute difference between consecutive frames.

    frames: [N, H, W, C] float in 0..1 (ComfyUI IMAGE). Returns a list of N
    values; entry 0 is 0. Works on a small grey copy, so it is cheap.
    """
    n, h, w, _ = frames.shape
    if n < 2:
        return [0.0] * n
    scale = max_side / max(h, w)
    size = (max(8, int(round(h * min(1.0, scale)))), max(8, int(round(w * min(1.0, scale)))))
    out = [0.0]
    prev = None
    for start in range(0, n, 64):
        x = frames[start:start + 64].to(torch.float32).mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=size, mode="area")
        if prev is not None:
            x = torch.cat([prev, x], dim=0)
        diff = (x[1:] - x[:-1]).abs().mean(dim=(1, 2, 3))
        out.extend(diff.tolist())
        prev = x[-1:]
    return out[:n]


def chunk_frame_ranges(num_chunks):
    """Input-frame ranges consumed by each denoising chunk (see the pipelines' LQ_video slicing)."""
    ranges = [(0, 25)]
    for idx in range(1, num_chunks):
        ranges.append((idx * 8 + 17, idx * 8 + 25))
    return ranges


class MotionBudget:
    def __init__(self, motion, num_chunks, intensity=0.5, dynamic_local=True, verbose=True):
        self.intensity = float(max(0.0, min(1.0, intensity)))
        self.dynamic_local = dynamic_local
        self.verbose = verbose
        # motion: one value per *real* input frame. Chunks that only cover the
        # padding (repeated last frame) reuse the previous chunk's score.
        self.scores = []
        n = len(motion)
        for lo, hi in chunk_frame_ranges(num_chunks):
            seg = motion[lo:min(hi, n)]
            if not seg:
                self.scores.append(self.scores[-1] if self.scores else 0.0)
                continue
            m = sum(seg) / len(seg)
            s = (m - MOTION_LOW) / (MOTION_HIGH - MOTION_LOW)
            self.scores.append(max(0.0, min(1.0, s)))

    def __call__(self, idx, topk_ratio, kv_ratio, local_range):
        if self.intensity <= 0.0 or idx >= len(self.scores):
            return topk_ratio, kv_ratio, local_range
        s = self.scores[idx]
        topk = topk_ratio * (1.0 + 0.25 * self.intensity * (2.0 * s - 1.0))
        kv = max(2.0, float(kv_ratio) - round(self.intensity * s))
        local = local_range
        if self.dynamic_local:
            local = 11 if s >= 0.5 else 9
        if self.verbose and (idx == 0 or idx % 4 == 3 or idx == len(self.scores) - 1):
            print(f"[FlashVSR-Linux] chunk {idx + 1}/{len(self.scores)} motion={s:.2f} "
                  f"topk_ratio={topk:.3f} kv={kv:.0f} local={local}", flush=True)
        return topk, kv, local
