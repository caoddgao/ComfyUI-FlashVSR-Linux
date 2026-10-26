"""CPU unit tests: attention masking, tiling, budget, filename prefix, ffmpeg writer, nodes.

    python tests/test_units.py
"""
import datetime
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

import _harness as h

attention = h.sub("runtime.attention")
tiling = h.sub("runtime.tiling")
budget = h.sub("runtime.budget")
video = h.sub("runtime.video")
nodes = h.sub("nodes")

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def test_sdpa_block_mask_matches_dense_reference():
    torch.manual_seed(0)
    b, heads, d = 1, 2, 64
    nq, nk = 5, 7
    q = torch.randn(b, nq * 128, heads * d)
    k = torch.randn(b, nk * 128, heads * d)
    v = torch.randn(b, nk * 128, heads * d)
    mask = torch.rand(b, heads, nq, nk) > 0.5
    mask[..., 0] = True  # every query window sees something
    out = attention._sdpa_block_masked(q, k, v, mask, heads, q_windows_per_step=2)
    # dense reference: explicit softmax with -inf outside the mask
    qh = q.view(b, -1, heads, d).transpose(1, 2)
    kh = k.view(b, -1, heads, d).transpose(1, 2)
    vh = v.view(b, -1, heads, d).transpose(1, 2)
    token_mask = mask.repeat_interleave(128, 2).repeat_interleave(128, 3)
    scores = (qh @ kh.transpose(-1, -2)) / d ** 0.5
    scores = scores.masked_fill(~token_mask, float("-inf"))
    ref = (scores.softmax(-1) @ vh).transpose(1, 2).reshape(b, -1, heads * d)
    err = (out - ref).abs().max().item()
    report("sdpa block-masked attention == dense masked softmax", err < 1e-4, f"max_abs_err={err:.2e}")


def test_empty_query_window_gives_zeros():
    q = torch.randn(1, 256, 128)
    k = torch.randn(1, 256, 128)
    v = torch.randn(1, 256, 128)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    mask[0, 0, 0, 1] = True
    out = attention._sdpa_block_masked(q, k, v, mask, 1)
    ok = torch.isfinite(out).all().item() and out[0, 128:].abs().max().item() == 0
    report("query window with no keys -> finite zeros", ok)


def test_backend_resolution_on_cpu():
    got = attention.resolve_backend("sparse_sage2", "cpu")
    report("sparse backends fall back to sdpa on CPU", got == "sdpa", f"got={got}")


def test_dispatch_uses_selected_backend():
    calls = []
    original = attention._sdpa_block_masked
    attention._sdpa_block_masked = lambda *a, **kw: calls.append("sdpa") or original(*a, **kw)
    try:
        q = torch.randn(1, 128, 128)
        with attention.use_backend("sdpa"):
            attention.sparse_attention(q, q, q, torch.ones(1, 1, 1, 1, dtype=torch.bool), 1)
        inner = attention.current_backend()
    finally:
        attention._sdpa_block_masked = original
    report("use_backend() selects the backend and restores the default", calls == ["sdpa"] and inner == "sparse_sage2")


def test_tiling_covers_frame_with_overlap():
    ok = True
    worst = None
    for out_h, out_w, tile, overlap in [(384, 512, 128, 32), (1024, 1920 // 128 * 128 + 128, 1024, 96),
                                        (128, 128, 1024, 96), (640, 384, 256, 64)]:
        tiles, overlap = tiling.plan(out_h, out_w, tile, overlap)
        cover = torch.zeros(out_h, out_w)
        wsum = torch.zeros(out_h, out_w)
        for y0, x0, th, tw in tiles:
            ok &= th % 128 == 0 and tw % 128 == 0 and y0 >= 0 and x0 >= 0 and y0 + th <= out_h and x0 + tw <= out_w
            cover[y0:y0 + th, x0:x0 + tw] += 1
            wsum[y0:y0 + th, x0:x0 + tw] += tiling.blend_weight(th, tw, y0, x0, out_h, out_w, overlap)
        ok &= bool((cover > 0).all()) and bool((wsum > 0).all())
        ys = sorted({t[0] for t in tiles})
        for a, b in zip(ys, ys[1:]):
            got = (a + tiles[0][2]) - b
            worst = got if worst is None else min(worst, got)
            ok &= got >= overlap
    report("tiles are 128-multiples, cover the frame, overlap >= requested", ok, f"min_overlap={worst}")


def test_budget():
    motion = [0.0] + [0.1] * 30 + [0.0] * 30    # moving first half, static second half
    sched = budget.MotionBudget(motion, num_chunks=7, intensity=1.0, verbose=False)
    first = sched(0, 2.0, 3.0, 11)
    last = sched(6, 2.0, 3.0, 11)
    off = budget.MotionBudget(motion, 7, intensity=0.0, verbose=False)(0, 2.0, 3.0, 11)
    ok = first == (2.5, 2.0, 11) and last == (1.5, 3.0, 9) and off == (2.0, 3.0, 11)
    report("motion budget: moving -> more topk/local 11, static -> less/local 9, intensity 0 -> upstream", ok,
           f"first={first} last={last} off={off}")
    frames = torch.zeros(4, 32, 32, 3)
    frames[2:] = 1.0
    m = budget.frame_motion(frames)
    report("frame_motion detects a cut", m[0] == 0 and m[1] == 0 and abs(m[2] - 1) < 1e-6 and m[3] == 0, f"{m}")


def test_prefix():
    now = datetime.datetime(2026, 8, 31, 15, 42, 30, 123000)
    got = video.expand_prefix("FlashVSR/%date:yyyy/MM/dd%/up_%width%x%height%_%date:HHmmss.fff%_%year%", 640, 360, now)
    want = "FlashVSR/2026/08/31/up_640x360_154230.123_2026"
    report("filename prefix variables", got == want, got)
    try:
        video.next_path(h.TMP, "../escape")
        report("prefix cannot leave the output folder", False)
    except ValueError:
        report("prefix cannot leave the output folder", True)


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,nb_frames",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


def test_video_combine_node():
    node = nodes.FlashVSRLinuxVideoCombine()
    images = torch.rand(30, 120, 160, 3)
    audio = {"waveform": torch.sin(torch.linspace(0, 2000, 48000)).repeat(1, 2, 1) * 0.3, "sample_rate": 48000}
    res = node.combine(images, 24.0, "sub/%date:yyyyMMdd%/clip_%width%x%height%", "auto", 19, 3, True, audio=audio)
    path = res["result"][0]
    streams = ffprobe(path)
    v = [s for s in streams if s["codec_type"] == "video"][0]
    a = [s for s in streams if s["codec_type"] == "audio"]
    ok = (os.path.isfile(path) and v["width"] == 160 and v["height"] == 120 and int(v["nb_frames"]) == 30
          and len(a) == 1 and res["ui"]["flashvsr_video"][0]["type"] == "output"
          and len([f for f in os.listdir(os.path.dirname(path))]) == 1)
    report("video combine: one mp4 with video + audio", ok, f"{os.path.relpath(path, h.TMP)} {streams}")
    res2 = node.combine(images[:5], 24.0, "sub/%date:yyyyMMdd%/clip_%width%x%height%", "libx264", 23, 1, True)
    report("video combine: counter increments", res2["result"][0].endswith("_00002.mp4"), os.path.basename(res2["result"][0]))
    odd = node.combine(torch.rand(3, 61, 81, 3), 30.0, "odd", "libx264", 23, 2, False)
    s = ffprobe(odd["result"][0])[0]
    report("video combine: odd size cropped to even for yuv420p, temp output",
           (s["width"], s["height"]) == (80, 60) and odd["ui"]["flashvsr_video"][0]["type"] == "temp", str(s))


def test_settings_and_model_nodes():
    settings = nodes.FlashVSRLinuxSettings()
    req = settings.INPUT_TYPES()["required"]
    built = settings.build(**{k: (v[1]["default"] if len(v) > 1 and "default" in v[1] else v[0][0]) for k, v in req.items()})[0]
    runner = h.sub("runtime.runner")
    ok = set(built) == set(runner.DEFAULT_SETTINGS)
    report("settings node produces every runner setting", ok, str(sorted(set(built) ^ set(runner.DEFAULT_SETTINGS))))
    try:
        nodes.FlashVSRLinuxModel().load("FlashVSR-v1.1", "tiny", "bf16", "cpu")
        report("model node reports missing weight files", False)
    except FileNotFoundError as e:
        report("model node reports missing weight files", "diffusion_pytorch_model_streaming_dmd.safetensors" in str(e), str(e))


class _IdentityPipe:
    """Stands in for a FlashVSR pipeline: returns its LQ input (minus the 4 tail frames, like upstream)."""

    def __call__(self, LQ_video=None, num_frames=None, progress_bar_cmd=None, **kwargs):
        for _ in progress_bar_cmd(range((num_frames - 1) // 8 - 2)):
            pass
        return LQ_video[0, :, : num_frames - 4].float()

    def load_models_to_device(self, names):
        pass


class _IdentityModel:
    def __init__(self):
        self.pipe = _IdentityPipe()
        self.mode = "tiny"
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def to_compute(self):
        pass

    def offload_dit(self):
        pass

    def offload_all(self):
        pass


def test_run_geometry_with_identity_pipeline():
    runner = h.sub("runtime.runner")
    torch.manual_seed(0)
    frames = torch.rand(11, 45, 70, 3)
    for scale in (2, 4):
        ref = F.interpolate(frames.permute(0, 3, 1, 2), scale_factor=scale, mode="bicubic",
                            align_corners=False).clamp(0, 1).permute(0, 2, 3, 1)
        for strategy, tile in (("full_frame", 256), ("adaptive_tiles", 40), ("adaptive_tiles", 64)):
            out = runner.run(_IdentityModel(), frames, scale=scale, seed=0, color_fix=False,
                             settings={"spatial_strategy": strategy, "max_tile_edge": tile, "blend_overlap": 12,
                                       "intensity": 0.0, "attention_backend": "sdpa"})
            err = (out - ref).abs().max().item()
            report(f"run(): crop/pad/blend/frame trim exact, x{scale} {strategy} tile={tile}",
                   out.shape == ref.shape and err < 1e-5, f"shape={tuple(out.shape)} max_abs_err={err:.1e}")


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except Exception as error:
                report(name, False, f"raised {type(error).__name__}: {error}")
    print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
