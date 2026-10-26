"""End-to-end CPU smoke test with small, randomly initialised models.

This checks shapes, frame counts, tiling, the budget hook and the attention
dispatch through the real FlashVSR pipelines. Random weights mean the pixels
are meaningless; image quality can only be checked with the real weights on a GPU.

    python tests/test_cpu_pipeline.py
"""
import sys
import time

import torch

import _harness as h

runner = h.sub("runtime.runner")
attention = h.sub("runtime.attention")
WanModel = h.sub("flashvsr.models.wan_video_dit").WanModel
WanVideoVAE = h.sub("flashvsr.models.wan_video_vae").WanVideoVAE
lq_proj = h.sub("flashvsr.models.lq_proj")
build_tcdecoder = h.sub("flashvsr.models.TCDecoder").build_tcdecoder

DIM = 256  # 2 heads x head_dim 128 (Wan RoPE needs head_dim 128)
DTYPE = torch.float32


def small_model(mode, v11=True):
    torch.manual_seed(0)
    dit = WanModel(dim=DIM, in_dim=16, ffn_dim=256, out_dim=16, text_dim=4096, freq_dim=256, eps=1e-6,
                   patch_size=(1, 2, 2), num_heads=2, num_layers=2)
    pipe = runner.PIPELINES[mode](device="cpu", torch_dtype=DTYPE)
    pipe.dit = dit
    pipe.vae = WanVideoVAE() if mode == "full" else None
    proj_cls = lq_proj.Causal_LQ4x_Proj if v11 else lq_proj.Buffer_LQ4x_Proj
    proj = proj_cls(in_dim=3, out_dim=DIM, layer_num=1)
    tcd = None if mode == "full" else build_tcdecoder(new_channels=[32, 16, 16, 16], device="cpu", dtype=DTYPE,
                                                       new_latent_channels=16 + 768)
    ctx = torch.load(runner.BUNDLED_PROMPT, map_location="cpu").to(DTYPE)
    return runner.assemble(pipe, mode, proj, tcd, "cpu", DTYPE, context=ctx)


def check(model, n, h_in, w_in, scale, settings, label):
    frames = torch.rand(n, h_in, w_in, 3)
    calls = []
    t = time.time()
    out = runner.run(model, frames, scale=scale, settings=settings, seed=1, color_fix=True,
                     progress=lambda d, tot: calls.append((d, tot)))
    ok = (tuple(out.shape) == (n, h_in * scale, w_in * scale, 3) and torch.isfinite(out).all().item()
          and out.min().item() >= 0 and out.max().item() <= 1 and calls and calls[-1][0] == calls[-1][1])
    print(f"{'PASS' if ok else 'FAIL'} {label}: in {n}x{h_in}x{w_in} x{scale} -> {tuple(out.shape)} "
          f"range=({out.min().item():.3f},{out.max().item():.3f}) progress={calls[-1] if calls else None} "
          f"{time.time() - t:.1f}s", flush=True)
    return ok


def main():
    results = []
    base = {"attention_backend": "sdpa", "memory_policy": "resident"}
    tiny = small_model("tiny")
    results.append(check(tiny, 13, 32, 40, 4, dict(base, spatial_strategy="full_frame", intensity=0.0),
                         "tiny full_frame, 13 frames (padded to 25), static budget"))
    results.append(check(tiny, 30, 32, 32, 4, dict(base, spatial_strategy="full_frame", intensity=1.0),
                         "tiny full_frame, 30 frames, motion budget"))
    results.append(check(tiny, 1, 32, 32, 2, dict(base, spatial_strategy="full_frame"),
                         "tiny single image x2"))
    results.append(check(tiny, 9, 48, 72, 4, dict(base, spatial_strategy="adaptive_tiles", max_tile_edge=32,
                                                   blend_overlap=8, memory_policy="staged"),
                         "tiny adaptive_tiles (6 tiles), staged"))
    del tiny
    long = small_model("tiny-long")
    results.append(check(long, 40, 32, 32, 4, dict(base, spatial_strategy="full_frame"),
                         "tiny-long full_frame, 40 frames"))
    del long
    v10 = small_model("tiny", v11=False)
    results.append(check(v10, 21, 32, 32, 4, dict(base, spatial_strategy="full_frame"),
                         "tiny with v1.0 Buffer_LQ4x_Proj"))
    del v10
    full = small_model("full")
    results.append(check(full, 9, 32, 32, 4, dict(base, spatial_strategy="full_frame"),
                         "full (Wan VAE decode), resident"))
    results.append(check(full, 9, 32, 32, 4, dict(base, spatial_strategy="full_frame", memory_policy="staged"),
                         "full (tiled Wan VAE decode), staged"))
    print(f"{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
