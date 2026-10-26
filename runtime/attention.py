"""Sparse attention dispatch for FlashVSR's locality-constrained sparse attention (LCSA).

The FlashVSR DiT builds a block mask over 128-token windows
(``generate_draft_block_mask`` in ``flashvsr/models/wan_video_dit.py``):
``mask[b, h, i, j]`` is True when query window ``i`` may attend to key window ``j``.
Upstream FlashVSR feeds that mask to ``block_sparse_attn``. This module lets the
same mask run on one of several backends:

* ``sparse_sage2``      -- SpargeAttn ``block_sparse_sage2_attn_cuda`` (INT8 QK / FP8 PV)
* ``block_sparse_attn`` -- mit-han-lab Block-Sparse-Attention (the upstream FlashVSR kernel)
* ``sdpa``              -- PyTorch scaled_dot_product_attention with the expanded mask
                           (slow reference path; also works on CPU)

All functions take ``q, k, v`` shaped ``[B, L, H*D]`` like the rest of the DiT.
"""

import threading

import torch
import torch.nn.functional as F
from einops import rearrange

WINDOW_TOKENS = 128  # 2 x 8 x 8 window used by FlashVSR's SelfAttention

BACKENDS = ("sparse_sage2", "block_sparse_attn", "sdpa")

_state = threading.local()


def _probe(name):
    try:
        if name == "sparse_sage2":
            from spas_sage_attn import block_sparse_sage2_attn_cuda  # noqa: F401
        elif name == "block_sparse_attn":
            # torch is already imported above; block_sparse_attn_cuda needs libc10 loaded first
            from block_sparse_attn import block_sparse_attn_func  # noqa: F401
        return True, ""
    except Exception as error:  # ImportError, missing CUDA libs, ABI mismatch, ...
        return False, f"{type(error).__name__}: {error}"


_available_cache = {}


def backend_available(name):
    """Return (available, reason) for an attention backend."""
    if name == "sdpa":
        return True, ""
    if name not in _available_cache:
        _available_cache[name] = _probe(name)
    return _available_cache[name]


def resolve_backend(name, device):
    """Pick a usable backend, falling back with a printed reason."""
    if name not in BACKENDS:
        raise ValueError(f"unknown attention backend {name!r}, expected one of {BACKENDS}")
    if torch.device(device).type != "cuda":
        if name != "sdpa":
            print(f"[FlashVSR-Linux] {name} needs CUDA; using sdpa on {device}", flush=True)
        return "sdpa"
    order = [name] + [b for b in ("sparse_sage2", "block_sparse_attn") if b != name] + ["sdpa"]
    for candidate in order:
        ok, reason = backend_available(candidate)
        if ok:
            if candidate != name:
                print(f"[FlashVSR-Linux] attention backend {name} unavailable, using {candidate}", flush=True)
            return candidate
        if candidate == name:
            print(f"[FlashVSR-Linux] {name} unavailable: {reason}", flush=True)
    return "sdpa"


class use_backend:
    """Context manager selecting the sparse attention backend for the current thread."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.previous = getattr(_state, "backend", None)
        _state.backend = self.name
        return self

    def __exit__(self, *exc):
        _state.backend = self.previous
        return False


def current_backend():
    return getattr(_state, "backend", None) or "sparse_sage2"


def sparse_attention(q, k, v, block_mask, num_heads):
    """Masked attention over 128-token windows. q/k/v: [B, L, H*D]; block_mask: [B, H, Lq/128, Lk/128] bool."""
    backend = current_backend()
    if backend == "sparse_sage2":
        return _sparse_sage2(q, k, v, block_mask, num_heads)
    if backend == "block_sparse_attn":
        return _block_sparse_attn(q, k, v, block_mask, num_heads)
    return _sdpa_block_masked(q, k, v, block_mask, num_heads)


def _sparse_sage2(q, k, v, block_mask, num_heads):
    from spas_sage_attn import block_sparse_sage2_attn_cuda

    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    # SpargeAttn quantizes Q in 128-token blocks and K in 64-token blocks
    # (64 / 128 on sm90), so each 128x128 window entry covers two K (or Q) blocks.
    major, _ = torch.cuda.get_device_capability(q.device)
    if major == 9:
        mask_id = block_mask.repeat_interleave(2, dim=-2)
    else:
        mask_id = block_mask.repeat_interleave(2, dim=-1)
    mask_id = mask_id.to(torch.int8).contiguous()
    out_dtype = q.dtype
    x = block_sparse_sage2_attn_cuda(q, k, v, mask_id=mask_id, tensor_layout="HND")
    return rearrange(x.to(out_dtype), "b n s d -> b s (n d)", n=num_heads)


def _block_sparse_attn(q, k, v, block_mask, num_heads):
    # Same call as upstream FlashVSR's flash_attention().
    from block_sparse_attn import block_sparse_attn_func

    seqlen = q.shape[1]
    seqlen_kv = k.shape[1]
    q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
    k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
    v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
    cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
    head_mask_type = torch.tensor([1] * num_heads, device=q.device, dtype=torch.int32)
    x = block_sparse_attn_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type,
        None,
        block_mask,
        seqlen, seqlen_kv,
        0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    ).unsqueeze(0)
    return rearrange(x, "b s n d -> b s (n d)", n=num_heads)


def _sdpa_block_masked(q, k, v, block_mask, num_heads, q_windows_per_step=8):
    """Reference implementation: dense SDPA with the block mask expanded to tokens.

    Processes a few query windows at a time so the expanded mask stays small.
    """
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    lq = q.shape[2]
    out = torch.empty_like(q)
    n_qw = block_mask.shape[2]
    for start in range(0, n_qw, q_windows_per_step):
        end = min(n_qw, start + q_windows_per_step)
        blk = block_mask[:, :, start:end, :]
        token_mask = blk.repeat_interleave(WINDOW_TOKENS, dim=2).repeat_interleave(WINDOW_TOKENS, dim=3)
        token_mask = token_mask[:, :, :, : k.shape[2]]
        q_slice = q[:, :, start * WINDOW_TOKENS: min(lq, end * WINDOW_TOKENS)]
        token_mask = token_mask[:, :, : q_slice.shape[2]]
        o = F.scaled_dot_product_attention(q_slice, k, v, attn_mask=token_mask)
        # A query window with no allowed key window gives NaN; treat it as zero output.
        out[:, :, start * WINDOW_TOKENS: start * WINDOW_TOKENS + q_slice.shape[2]] = torch.nan_to_num(o)
    return rearrange(out, "b n s d -> b s (n d)", n=num_heads)
