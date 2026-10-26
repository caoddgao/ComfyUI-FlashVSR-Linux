"""Model loading and the end-to-end FlashVSR run used by the ComfyUI nodes.

Kept free of ComfyUI imports so it can be tested on its own.
Model loading follows upstream FlashVSR's examples/WanVSR/infer_flashvsr*_*.py.
"""

import gc
import os

import torch
import torch.nn.functional as F

from ..flashvsr.models import ModelManager
from ..flashvsr.models.lq_proj import Buffer_LQ4x_Proj, Causal_LQ4x_Proj
from ..flashvsr.models.TCDecoder import build_tcdecoder
from ..flashvsr.pipelines import FlashVSRFullPipeline, FlashVSRTinyLongPipeline, FlashVSRTinyPipeline
from . import attention, budget, tiling

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLED_PROMPT = os.path.join(PLUGIN_DIR, "assets", "posi_prompt.pth")

DIT_FILE = "diffusion_pytorch_model_streaming_dmd.safetensors"
LQ_PROJ_FILE = "LQ_proj_in.ckpt"
TCDECODER_FILE = "TCDecoder.ckpt"
VAE_FILE = "Wan2.1_VAE.pth"
PROMPT_FILE = "posi_prompt.pth"

PIPELINES = {"tiny": FlashVSRTinyPipeline, "tiny-long": FlashVSRTinyLongPipeline, "full": FlashVSRFullPipeline}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# First chunk needs 25 input frames (8 * 3 + 1), see process_total_num in the pipelines.
MIN_PIPELINE_FRAMES = 25
# Upstream pads 4 frames at the end and keeps (num_frames - 4) outputs.
TAIL_PAD = 4


def check_model_dir(model_dir, mode):
    need = [DIT_FILE, LQ_PROJ_FILE] + ([VAE_FILE] if mode == "full" else [TCDECODER_FILE])
    missing = [f for f in need if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        raise FileNotFoundError(f"{model_dir} is missing {', '.join(missing)}")


def is_v11(model_dir):
    """FlashVSR v1.1 uses the causal LQ projection, v1.0 the buffered one (upstream examples)."""
    return "1.1" in os.path.basename(os.path.normpath(model_dir))


class FlashVSRModel:
    """A loaded FlashVSR pipeline plus what the runner needs to drive it."""

    def __init__(self, pipe, mode, device, dtype):
        self.pipe = pipe
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype

    # -- memory placement -------------------------------------------------
    def to_compute(self):
        pipe = self.pipe
        pipe.load_models_to_device(["dit", "vae"] if self.mode == "full" else ["dit"])
        pipe.dit.LQ_proj_in.to(self.device)
        if getattr(pipe, "TCDecoder", None) is not None:
            pipe.TCDecoder.to(self.device)

    def offload_dit(self):
        """Move the DiT off the GPU before decoding (memory_policy=staged)."""
        pipe = self.pipe
        pipe.load_models_to_device(["vae"] if self.mode == "full" else [])
        pipe.dit.LQ_proj_in.to("cpu")
        empty_cache()

    def offload_all(self):
        pipe = self.pipe
        pipe.load_models_to_device([])
        pipe.dit.LQ_proj_in.to("cpu")
        if getattr(pipe, "TCDecoder", None) is not None:
            pipe.TCDecoder.to("cpu")
        empty_cache()


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_dir, mode="tiny", precision="bf16", device="cuda", prompt_path=None):
    if mode not in PIPELINES:
        raise ValueError(f"mode must be one of {list(PIPELINES)}")
    check_model_dir(model_dir, mode)
    dtype = DTYPES[precision]
    device = torch.device(device)

    mm = ModelManager(torch_dtype=dtype, device="cpu")
    files = [os.path.join(model_dir, DIT_FILE)]
    if mode == "full":
        files.append(os.path.join(model_dir, VAE_FILE))
    mm.load_models(files)
    pipe = PIPELINES[mode].from_model_manager(mm, device=str(device))

    proj_cls = Causal_LQ4x_Proj if is_v11(model_dir) else Buffer_LQ4x_Proj
    lq_proj = proj_cls(in_dim=3, out_dim=1536, layer_num=1)
    lq_proj.load_state_dict(torch.load(os.path.join(model_dir, LQ_PROJ_FILE), map_location="cpu"), strict=True)

    tcdecoder = None
    if mode != "full":
        tcdecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device="cpu", dtype=dtype,
                                    new_latent_channels=16 + 768)
        missing = tcdecoder.load_state_dict(torch.load(os.path.join(model_dir, TCDECODER_FILE), map_location="cpu"),
                                            strict=False)
        print(f"[FlashVSR-Linux] TCDecoder: {missing}", flush=True)

    if prompt_path is None:
        own = os.path.join(model_dir, PROMPT_FILE)
        prompt_path = own if os.path.isfile(own) else BUNDLED_PROMPT
    return assemble(pipe, mode, lq_proj, tcdecoder, device, dtype, prompt_path)


def assemble(pipe, mode, lq_proj, tcdecoder, device, dtype, prompt_path=None, context=None):
    """Finish a pipeline the way upstream's init_pipeline() does. Also used by the tests."""
    pipe.dit.LQ_proj_in = lq_proj.to(device, dtype=dtype)
    if mode == "full":
        # upstream: the encoder is never used (the LQ projection replaces it)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        pipe.TCDecoder = tcdecoder.to(device, dtype=dtype)
    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    if context is None:
        context = torch.load(prompt_path, map_location="cpu")
    pipe.init_cross_kv(context_tensor=context)
    model = FlashVSRModel(pipe, mode, device, dtype)
    model.to_compute()
    return model


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "quality_profile": "balanced",
    "intensity": 0.5,
    "spatial_strategy": "auto",
    "memory_policy": "auto",
    "attention_budget": -1.0,   # sparse_ratio; <= 0 means "from quality_profile"
    "kv_retention": -1.0,       # kv_ratio;     <= 0 means "from quality_profile"
    "local_radius": "auto",     # "auto" (profile + dynamic), "9" or "11"
    "max_tile_edge": 256,       # input pixels
    "blend_overlap": 24,        # input pixels
    "preprocess_batch": 8,
    "attention_backend": "sparse_sage2",
}


def pipeline_frame_count(n):
    """Frames fed to the pipeline for n input frames: 8k+1, >= n + TAIL_PAD and >= 25."""
    need = max(n + TAIL_PAD, MIN_PIPELINE_FRAMES)
    return (need - 1 + 7) // 8 * 8 + 1


def free_vram(device):
    if device.type != "cuda":
        return None
    free, _ = torch.cuda.mem_get_info(device)
    return free


def choose_tiles(settings, in_h, in_w, scale, pad_h, pad_w, device):
    strategy = settings["spatial_strategy"]
    if strategy == "auto":
        # Full frame up to ~1080p output when the GPU has room, tiles otherwise.
        free = free_vram(device)
        big_gpu = free is None or free >= 20 * 1024 ** 3
        strategy = "full_frame" if (pad_h * pad_w <= 1920 * 1088 and big_gpu) else "adaptive_tiles"
    if strategy == "full_frame":
        return [(0, 0, pad_h, pad_w)], 0
    return tiling.plan(pad_h, pad_w, settings["max_tile_edge"] * scale, settings["blend_overlap"] * scale)


def resolve_params(settings):
    sparse, kv, local = budget.PROFILES[settings["quality_profile"]]
    if settings["attention_budget"] and settings["attention_budget"] > 0:
        sparse = float(settings["attention_budget"])
    if settings["kv_retention"] and settings["kv_retention"] > 0:
        kv = float(settings["kv_retention"])
    pinned_local = str(settings["local_radius"]) in ("9", "11")
    if pinned_local:
        local = int(settings["local_radius"])
    return sparse, kv, local, pinned_local


@torch.no_grad()
def upscale_tile(frames, idx, scale, out_h, out_w, pad_h, pad_w, tile, device, dtype, batch):
    """Bicubic-upscale frames[idx] to the output size, pad to the 128 grid, crop one tile.

    Returns [1, 3, len(idx), th, tw] in [-1, 1] (FlashVSR's LQ_video format).
    """
    y0, x0, th, tw = tile
    parts = []
    for start in range(0, len(idx), batch):
        sel = idx[start:start + batch]
        x = frames[sel].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(out_h, out_w), mode="bicubic", align_corners=False)
        if pad_h != out_h or pad_w != out_w:
            x = F.pad(x, (0, pad_w - out_w, 0, pad_h - out_h), mode="replicate")
        x = x[:, :, y0:y0 + th, x0:x0 + tw].clamp_(0, 1).mul_(2).sub_(1)
        parts.append(x.to(dtype))
    return torch.cat(parts, 0).permute(1, 0, 2, 3).unsqueeze(0)


def run(model, frames, scale=4, settings=None, seed=0, color_fix=True, progress=None):
    """Upscale a ComfyUI IMAGE batch [N, H, W, 3] (0..1) and return [N, H*scale, W*scale, 3] on CPU."""
    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    pipe = model.pipe
    device = model.device
    n, in_h, in_w, _ = frames.shape
    if n < 1:
        raise ValueError("no input frames")
    frames = frames.cpu()

    out_h, out_w = in_h * scale, in_w * scale
    pad_h, pad_w = tiling.round_up(out_h), tiling.round_up(out_w)
    num_frames = pipeline_frame_count(n)
    idx = list(range(n)) + [n - 1] * (num_frames - n)

    tiles, overlap = choose_tiles(s, in_h, in_w, scale, pad_h, pad_w, device)
    backend = s["attention_backend"]
    if backend == "auto":
        backend = "block_sparse_attn" if len(tiles) == 1 else "sparse_sage2"
    backend = attention.resolve_backend(backend, device)

    memory = s["memory_policy"]
    if memory == "auto":
        free = free_vram(device)
        memory = "staged" if (free is not None and free < 24 * 1024 ** 3) or len(tiles) > 1 else "resident"
    staged = memory == "staged"

    sparse, kv, local, pinned_local = resolve_params(s)
    num_chunks = (num_frames - 1) // 8 - 2
    schedule = None
    if s["intensity"] > 0:
        motion = budget.frame_motion(frames)  # real frames only, not the padding
        schedule = budget.MotionBudget(motion, num_chunks, intensity=s["intensity"], dynamic_local=not pinned_local)

    print(f"[FlashVSR-Linux] {n} frames {in_w}x{in_h} -> {out_w}x{out_h} (x{scale}), mode={model.mode}, "
          f"pipeline frames={num_frames}, tiles={len(tiles)} ({tiles[0][3]}x{tiles[0][2]}), "
          f"attention={backend}, memory={memory}, sparse_ratio={sparse}, kv={kv}, local={local}", flush=True)

    single = len(tiles) == 1
    result = torch.zeros((n, out_h, out_w, 3), dtype=torch.float32)
    weight_sum = None if single else torch.zeros((out_h, out_w), dtype=torch.float32)
    lq_device = torch.device("cpu") if model.mode == "tiny-long" else device

    total_steps = len(tiles) * num_chunks
    done = [0]

    def progress_bar(iterable):
        for item in iterable:
            yield item
            done[0] += 1
            if progress is not None:
                progress(done[0], total_steps)

    for tile in tiles:
        y0, x0, th, tw = tile
        lq = upscale_tile(frames, idx, scale, out_h, out_w, pad_h, pad_w, tile, device, model.dtype,
                          s["preprocess_batch"])
        lq = lq.to(lq_device)
        model.to_compute()
        with attention.use_backend(backend):
            video = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
                LQ_video=lq, num_frames=num_frames, height=th, width=tw,
                is_full_block=False, if_buffer=True,
                topk_ratio=sparse * 768 * 1280 / (th * tw),   # upstream: sparse_ratio scaled to 1280x768
                kv_ratio=kv, local_range=local,
                color_fix=color_fix,
                tiled=staged,   # full mode: tiled Wan VAE decode when memory is tight
                chunk_schedule=schedule,
                before_decode=model.offload_dit if staged else None,
                progress_bar_cmd=progress_bar,
            )
        del lq
        # video: [3, F_out, th, tw] in [-1, 1]
        got = video.shape[1]
        if got < n:
            raise RuntimeError(f"pipeline returned {got} frames for {n} input frames")
        tile_out = video[:, :n].float().add_(1).mul_(0.5).clamp_(0, 1).permute(1, 2, 3, 0).cpu()
        del video
        # crop the padding away (only the bottom/right tiles reach it)
        h_keep = max(0, min(th, out_h - y0))
        w_keep = max(0, min(tw, out_w - x0))
        tile_out = tile_out[:, :h_keep, :w_keep]
        if single:
            result[:, y0:y0 + h_keep, x0:x0 + w_keep] = tile_out
        else:
            wgt = tiling.blend_weight(th, tw, y0, x0, pad_h, pad_w, overlap)[:h_keep, :w_keep]
            result[:, y0:y0 + h_keep, x0:x0 + w_keep] += tile_out * wgt[None, :, :, None]
            weight_sum[y0:y0 + h_keep, x0:x0 + w_keep] += wgt
        del tile_out
        if staged:
            model.offload_all()
        empty_cache()

    if not single:
        result /= weight_sum.clamp_(min=1e-6)[None, :, :, None]
    if staged:
        model.offload_all()
    return result
