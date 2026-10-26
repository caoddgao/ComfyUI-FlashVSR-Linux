"""GPU self-check (run on the machine with the GPU, inside ComfyUI's venv):

    python tests/check_gpu.py                       # attention kernels vs. SDPA reference
    python tests/check_gpu.py --model-dir ~/comfyui/models/FlashVSR-v1.1 [--mode tiny]

1. Runs each sparse attention backend on FlashVSR-sized inputs with a FlashVSR-style
   block mask and compares it with the dense masked SDPA reference.
2. With --model-dir, loads the real weights and upscales a short synthetic clip.
"""
import argparse
import time

import torch

import _harness as h

attention = h.sub("runtime.attention")
runner = h.sub("runtime.runner")


def bench(fn, repeat=5):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(repeat):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.time() - t) / repeat * 1000


def flashvsr_like_mask(heads, nq, nk, topk_per_row, device):
    """Random top-k block mask; always keeps the diagonal-ish local windows like LCSA does."""
    scores = torch.rand(1, heads, nq, nk, device=device)
    local = torch.arange(nq, device=device)[:, None] * nk // nq
    scores.scatter_(3, local.expand(1, heads, nq, 1), 2.0)
    thresh = scores.topk(topk_per_row, dim=-1).values[..., -1:]
    return scores >= thresh


def check_attention(device):
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(device)} "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(device)))}")
    for name in ("sparse_sage2", "block_sparse_attn"):
        ok, reason = attention.backend_available(name)
        print(f"  {name:18s} {'available' if ok else 'NOT available: ' + reason}")
    heads, d = 12, 128                    # Wan2.1-1.3B
    nq, nk = 60, 240                      # one 2-latent-frame chunk at 1280x768 + 3 cached chunks
    torch.manual_seed(0)
    q = torch.randn(1, nq * 128, heads * d, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, nk * 128, heads * d, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, nk * 128, heads * d, device=device, dtype=torch.bfloat16)
    mask = flashvsr_like_mask(heads, nq, nk, topk_per_row=int(nk * 0.25), device=device)
    ref, t_ref = bench(lambda: attention._sdpa_block_masked(q, k, v, mask, heads), repeat=2)
    print(f"  sdpa reference      {t_ref:8.2f} ms   (density {mask.float().mean().item():.2f})")
    all_ok = True
    for name in ("sparse_sage2", "block_sparse_attn"):
        if not attention.backend_available(name)[0]:
            continue
        fn = attention._sparse_sage2 if name == "sparse_sage2" else attention._block_sparse_attn
        out, t = bench(lambda: fn(q, k, v, mask, heads))
        cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
        err = (out.float() - ref.float()).abs().mean().item()
        ok = cos > 0.99 and torch.isfinite(out).all().item()
        all_ok &= ok
        print(f"  {name:18s} {t:8.2f} ms   cosine={cos:.5f} mean_abs_err={err:.4f}  {'OK' if ok else 'MISMATCH'}")
    return all_ok


def check_model(model_dir, mode, device, frames, size):
    print(f"loading {model_dir} ({mode}) ...")
    t = time.time()
    model = runner.load_model(model_dir, mode=mode, precision="bf16", device=device)
    print(f"  loaded in {time.time() - t:.1f}s")
    hh, ww = size
    yy, xx = torch.meshgrid(torch.linspace(0, 1, hh), torch.linspace(0, 1, ww), indexing="ij")
    clip = torch.stack([torch.stack([(xx + i / frames) % 1, yy, (xx * yy + i / 7) % 1], -1) for i in range(frames)])
    for backend in ("sparse_sage2", "block_sparse_attn"):
        if not attention.backend_available(backend)[0]:
            continue
        torch.cuda.reset_peak_memory_stats(device)
        t = time.time()
        out = runner.run(model, clip, scale=4, settings={"attention_backend": backend, "spatial_strategy": "full_frame"})
        dt = time.time() - t
        print(f"  {backend:18s} {tuple(clip.shape)} -> {tuple(out.shape)} in {dt:.1f}s "
              f"({frames / dt:.2f} fps), peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB, "
              f"finite={torch.isfinite(out).all().item()} mean={out.mean().item():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-dir")
    ap.add_argument("--mode", default="tiny", choices=["tiny", "tiny-long", "full"])
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--size", type=int, nargs=2, default=[192, 320], metavar=("H", "W"))
    args = ap.parse_args()
    device = torch.device(args.device)
    ok = check_attention(device)
    if args.model_dir:
        check_model(args.model_dir, args.mode, device, args.frames, args.size)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
