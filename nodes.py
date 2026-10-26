import os

import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .runtime import runner, video

CATEGORY = "FlashVSR-Linux"
MODEL_DIR_NAMES = ["FlashVSR-v1.1", "FlashVSR"]
FOLDER_KEY = "flashvsr"

# Lets extra_model_paths.yaml use "flashvsr: <dir>" (a model folder or a folder containing them).
folder_paths.add_model_folder_path(FOLDER_KEY, os.path.join(folder_paths.models_dir, "FlashVSR-v1.1"))


def _is_model_dir(path):
    return os.path.isfile(os.path.join(path, runner.DIT_FILE))


def _subdirs(root):
    try:
        return [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, n))]
    except OSError:
        return []


def find_model_dirs():
    """{display name: absolute path} of FlashVSR model folders.

    Looks at the "flashvsr" entries of extra_model_paths.yaml (each may be a model
    folder or contain model folders) and at sub folders named *flashvsr* of
    ComfyUI/models and of the parent of every configured model folder
    (e.g. base_path in extra_model_paths.yaml).
    """
    found = []
    try:
        registered = folder_paths.get_folder_paths(FOLDER_KEY)
    except KeyError:
        registered = []
    for path in registered:
        if _is_model_dir(path):
            found.append(path)
        else:
            found += [d for d in _subdirs(path) if _is_model_dir(d)]
    roots = [folder_paths.models_dir]
    for paths, _ in folder_paths.folder_names_and_paths.values():
        roots += [os.path.dirname(os.path.normpath(p)) for p in paths]
    for root in dict.fromkeys(roots):
        found += [d for d in _subdirs(root) if "flashvsr" in os.path.basename(d).lower()]
    choices = {}
    for path in dict.fromkeys(os.path.abspath(p) for p in found):
        name = os.path.basename(path)
        if name in choices and choices[name] != path:
            name = f"{name} ({os.path.dirname(path)})"
        choices.setdefault(name, path)
    return choices


def model_dir_choices():
    names = list(find_model_dirs())
    return names or list(MODEL_DIR_NAMES)


def device_choices():
    choices = ["auto"]
    if torch.cuda.is_available():
        choices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return choices + ["cpu"]


_MODEL_CACHE = {}


class FlashVSRLinuxModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (model_dir_choices(), {"tooltip": "Folder under ComfyUI/models/ with the FlashVSR weights."}),
            "mode": (["tiny", "tiny-long", "full"], {"default": "tiny",
                     "tooltip": "tiny / tiny-long decode with TCDecoder (fast); full uses Wan2.1_VAE.pth."}),
            "precision": (["bf16", "fp16"], {"default": "bf16"}),
            "device": (device_choices(), {"default": "auto"}),
        }}

    RETURN_TYPES = ("FLASHVSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model, mode, precision, device):
        model_dir = find_model_dirs().get(model, os.path.join(folder_paths.models_dir, model))
        dev = mm.get_torch_device() if device == "auto" else torch.device(device)
        key = (model_dir, mode, precision, str(dev))
        if key not in _MODEL_CACHE:
            for old in list(_MODEL_CACHE):   # keep one pipeline in memory
                _MODEL_CACHE.pop(old).offload_all()
            mm.unload_all_models()
            mm.soft_empty_cache()
            _MODEL_CACHE[key] = runner.load_model(model_dir, mode=mode, precision=precision, device=dev)
        return (_MODEL_CACHE[key],)


class FlashVSRLinuxSettings:
    @classmethod
    def INPUT_TYPES(cls):
        d = runner.DEFAULT_SETTINGS
        return {"required": {
            "quality_profile": (["detail", "balanced", "throughput"], {"default": d["quality_profile"],
                                "tooltip": "detail: sparse 2.0 / local 9; balanced: 2.0 / 11; throughput: 1.5 / 11"}),
            "intensity": ("FLOAT", {"default": d["intensity"], "min": 0.0, "max": 1.0, "step": 0.05,
                          "tooltip": "Motion-aware per-chunk budget. 0 = fixed upstream settings."}),
            "spatial_strategy": (["auto", "full_frame", "adaptive_tiles"], {"default": d["spatial_strategy"]}),
            "memory_policy": (["auto", "resident", "staged"], {"default": d["memory_policy"]}),
            "attention_budget": ("FLOAT", {"default": d["attention_budget"], "min": -1.0, "max": 4.0, "step": 0.1,
                                 "tooltip": "sparse_ratio (upstream: 1.5 faster, 2.0 more stable). -1 = from profile."}),
            "kv_retention": ("FLOAT", {"default": d["kv_retention"], "min": -1.0, "max": 8.0, "step": 1.0,
                             "tooltip": "kv_ratio: earlier chunks kept in the KV cache (upstream 3). -1 = from profile."}),
            "local_radius": (["auto", "9", "11"], {"default": d["local_radius"],
                             "tooltip": "9 = sharper, 11 = more stable. auto = profile, adjusted by motion."}),
            "max_tile_edge": ("INT", {"default": d["max_tile_edge"], "min": 64, "max": 2048, "step": 8,
                              "tooltip": "Tile edge in input pixels for adaptive_tiles."}),
            "blend_overlap": ("INT", {"default": d["blend_overlap"], "min": 0, "max": 256, "step": 2,
                              "tooltip": "Tile overlap in input pixels."}),
            "preprocess_batch": ("INT", {"default": d["preprocess_batch"], "min": 1, "max": 64}),
            "attention_backend": (["sparse_sage2", "block_sparse_attn", "auto", "sdpa"],
                                  {"default": d["attention_backend"],
                                   "tooltip": "auto: block_sparse_attn for full frame, sparse_sage2 for tiles. "
                                              "sdpa: slow dense reference."}),
        }}

    RETURN_TYPES = ("FLASHVSR_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, **kwargs):
        return (dict(kwargs),)


class FlashVSRLinuxUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FLASHVSR_MODEL",),
                "frames": ("IMAGE",),
                "scale": ([4, 2], {"default": 4}),
                "color_fix": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {"settings": ("FLASHVSR_SETTINGS",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, model, frames, scale, color_fix, seed, settings=None):
        bar = comfy.utils.ProgressBar(1)

        def progress(done, total):
            mm.throw_exception_if_processing_interrupted()
            bar.update_absolute(done, total)

        out = runner.run(model, frames, scale=int(scale), settings=settings, seed=seed,
                         color_fix=color_fix, progress=progress)
        return (out,)


class FlashVSRLinuxVideoCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "FlashVSR-Linux",
                                    "tooltip": "May contain sub folders, %date:yyyyMMdd_HHmmss%, %width%, %height%."}),
                "codec": (["auto", "h264_nvenc", "hevc_nvenc", "libx264", "libx265"], {"default": "auto"}),
                "quality": ("INT", {"default": 19, "min": 0, "max": 51,
                            "tooltip": "CRF / CQ: lower is better quality and larger files."}),
                "workers": ("INT", {"default": 3, "min": 1, "max": 16,
                            "tooltip": "Threads converting frames ahead of ffmpeg."}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    OUTPUT_NODE = True
    FUNCTION = "combine"
    CATEGORY = CATEGORY

    def combine(self, images, frame_rate, filename_prefix, codec, quality, workers, save_output, audio=None):
        n, h, w, _ = images.shape
        if save_output:
            base_dir, kind = folder_paths.get_output_directory(), "output"
        else:
            base_dir, kind = folder_paths.get_temp_directory(), "temp"
        prefix = video.expand_prefix(filename_prefix, w, h)
        path, subfolder, filename = video.next_path(base_dir, prefix)
        used = video.encode(images, path, frame_rate, codec=codec, quality=quality, audio=audio, workers=workers)
        print(f"[FlashVSR-Linux] wrote {path} ({n} frames, {w}x{h}, {used})", flush=True)
        return {"ui": {"flashvsr_video": [{"filename": filename, "subfolder": subfolder, "type": kind}]},
                "result": (path,)}


NODE_CLASS_MAPPINGS = {
    "FlashVSRLinuxModel": FlashVSRLinuxModel,
    "FlashVSRLinuxSettings": FlashVSRLinuxSettings,
    "FlashVSRLinuxUpscale": FlashVSRLinuxUpscale,
    "FlashVSRLinuxVideoCombine": FlashVSRLinuxVideoCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRLinuxModel": "FlashVSR Model (Linux)",
    "FlashVSRLinuxSettings": "FlashVSR Settings (Linux)",
    "FlashVSRLinuxUpscale": "FlashVSR Upscale (Linux)",
    "FlashVSRLinuxVideoCombine": "FlashVSR Video Combine (Linux)",
}
