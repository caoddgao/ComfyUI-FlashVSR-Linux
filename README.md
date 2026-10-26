# ComfyUI-FlashVSR-Linux

[中文](#中文) · [English](#english)

<a id="中文"></a>
## 中文

在 Linux 上运行 [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) 视频超分的 ComfyUI 节点，完全开源（Apache-2.0）。

### 由来

[TE-Speed-FlashVSR](https://github.com/tl2012tl/TE-Speed-FlashVSR) 是一个很好用的 FlashVSR 节点，但核心代码是
Windows 专用的闭源二进制（`nodes.pyd`、`te_runtime/*.pyd`），在 Linux 上无法加载。本项目参考它公开 README 里的
节点设计和参数，基于上游 FlashVSR 的开源代码，重新实现了一套能在 Linux 上运行的节点：

- **没有**使用 TE-Speed-FlashVSR 闭源二进制里的任何代码；动态预算、分块、视频合成都是独立实现，效果和速度
  不会和 TE 版完全一样。
- 与 TE-Speed-FlashVSR 和 FlashVSR 的作者没有关联，也没有得到他们的背书。
- 模型推理代码来自上游 FlashVSR（Apache-2.0），改动见 `NOTICE` 和 `patches/flashvsr-upstream-changes.diff`。

### 测试状态（请先看）

| 项目 | 状态 |
|---|---|
| CPU 端到端（缩小的随机权重模型，tiny / tiny-long / full、v1.0 / v1.1、分块、显存分阶段、动态预算） | ✅ `tests/test_cpu_pipeline.py` 8/8 |
| 单元测试（分块注意力 vs 暴力参考、分块拼接逐像素精确、文件名变量、ffmpeg 编码 + 音频、节点） | ✅ `tests/test_units.py` 20/20 |
| GPU 上 sparse_sage2 / block_sparse_attn 与参考实现数值一致 | ⏳ 待测：`tests/check_gpu.py` |
| 真实权重的画质与速度 | ⏳ 待测（开发环境没有 GPU，也下载不到权重） |

### 安装

1. 克隆到 ComfyUI 的 `custom_nodes`：

   ```bash
   cd ~/comfyui/custom_nodes
   git clone https://github.com/caoddgao/ComfyUI-FlashVSR-Linux.git
   ```

2. 安装稀疏注意力轮子（至少一个，必须和你的 torch / CUDA / GPU 匹配）。
   [Releases](https://github.com/caoddgao/ComfyUI-FlashVSR-Linux/releases) 里有为 torch 2.14.1+cu130、
   RTX 50 系（sm_120）编译好的轮子，编译过程和验收记录见
   [BUILD_REPORT](https://github.com/caoddgao/TE-Speed-FlashVSR/blob/claude/adoring-ramanujan-ytnixo/linux-build/BUILD_REPORT.md)。
   其他 GPU / torch 版本需要自己从源码编译
   （[SpargeAttn](https://github.com/woct0rdho/SpargeAttn)、
   [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)）。

   ```bash
   ~/comfyui/.venv/bin/python -m pip install --no-deps spas_sage_attn-0.1.0+cu130torch2.14.1-cp310-abi3-linux_x86_64.whl
   ~/comfyui/.venv/bin/python -m pip install --no-deps block_sparse_attn-0.0.2+cu130torch2.14.1-cp312-cp312-linux_x86_64.whl  # 可选
   ```

3. 需要 `ffmpeg`（`sudo apt install ffmpeg`）。`einops`、`safetensors`、`tqdm` 一般 ComfyUI 已经装好。

4. 模型放在 `ComfyUI/models/FlashVSR-v1.1/`（或 `FlashVSR/` 放 v1.0）：

   ```text
   diffusion_pytorch_model_streaming_dmd.safetensors
   LQ_proj_in.ckpt
   TCDecoder.ckpt          # tiny / tiny-long
   Wan2.1_VAE.pth          # full
   posi_prompt.pth         # 可选，没有就用插件自带的（来自上游仓库）
   ```

   目录名里带 `1.1` 的按 v1.1 加载（因果 LQ 投影），否则按 v1.0。

   模型不在 `ComfyUI/models/` 下也可以，节点会自动查找：
   - `extra_model_paths.yaml` 里配置的各个模型路径的上一级目录（通常就是 `base_path`）下，名字含 `flashvsr` 的文件夹；
   - `extra_model_paths.yaml` 里的 `flashvsr:` 条目，可以指向模型文件夹本身，也可以指向包含多个模型文件夹的目录，例如：

     ```yaml
     my_models:
         base_path: /mnt/data/AI_Hub/image_models
         checkpoints: checkpoints
         flashvsr: FlashVSR-v1.1
     ```

如果之前装了 TE-Speed-FlashVSR，它在 Linux 上本来就加载不了，可以移出 `custom_nodes`。两者的节点名不同，不会冲突。

### 节点

| 节点 | 作用 |
|---|---|
| **FlashVSR Model (Linux)** | 加载模型：`model`（模型目录）、`mode`（tiny / tiny-long / full）、`precision`（bf16 / fp16）、`device` |
| **FlashVSR Settings (Linux)** | 可选的推理设置，接到 Upscale 的 `settings` |
| **FlashVSR Upscale (Linux)** | 输入 `IMAGE` 帧批次，输出放大后的 `IMAGE`：`scale`（4 / 2）、`color_fix`、`seed` |
| **FlashVSR Video Combine (Linux)** | 用 ffmpeg 写 mp4，节点内预览；`codec` auto 会优先用 `h264_nvenc`，不可用时用 `libx264`；可接 `audio` |

典型连线：Load Video → FlashVSR Upscale (Linux) → FlashVSR Video Combine (Linux)，Model 节点接到 Upscale。

#### Settings 参数

- `quality_profile`：`detail`（sparse 2.0 / local 9）、`balanced`（2.0 / 11，默认）、`throughput`（1.5 / 11）。
  这些是上游推荐值：sparse_ratio 1.5 更快、2.0 更稳；local_range 9 更锐、11 更稳。
- `intensity`：运动感知动态预算的强度，0 = 与上游一样全程固定参数。算法见 `runtime/budget.py`：
  按低清输入的帧差估计每个时间块的运动量（0–1），静止块少保留一些 top-k 窗口（最多 −25%）、用 local 9；
  运动块多保留一些（最多 +25%）、用 local 11；当 `intensity × 运动量 > 0.5` 时再少保留一个历史 KV 块
  （默认 0.5 的强度下 KV 不变）。这是本项目自己的启发式规则，效果需要实测，觉得不好就设成 0。
- `spatial_strategy`：`auto`（输出不超过约 1920×1088 且显存 ≥ 20GB 时整帧，否则分块）、`full_frame`、`adaptive_tiles`。
- `memory_policy`：`resident`（模型常驻显存）、`staged`（解码前把 DiT 移回内存，full 模式用分块 VAE 解码）、
  `auto`（空闲显存 < 24GB 或分块时用 staged）。
- `attention_budget` / `kv_retention`：直接指定 sparse_ratio / kv_ratio，-1 表示用 profile 的值。
- `local_radius`：`auto` / `9` / `11`；指定 9 或 11 时动态预算不再改它。
- `max_tile_edge` / `blend_overlap`：分块边长和重叠（输入像素）。块的输出尺寸会取整到 128 的倍数。
- `preprocess_batch`：双三次放大时每批处理的帧数。
- `attention_backend`：`sparse_sage2`（默认，SpargeAttn）、`block_sparse_attn`（上游原版 kernel）、
  `auto`（整帧用 block_sparse_attn，分块用 sparse_sage2）、`sdpa`（很慢的稠密参考实现，只用于对比）。
  选中的库不可用时会自动退到另一个，并在控制台打印原因。

#### 和上游一致的行为

- 输入先双三次放大到目标尺寸，再补边到 128 的倍数，结果会裁回原尺寸（上游是中心裁剪，会丢掉边缘；这里不丢）。
- 帧数会用最后一帧补齐到 8k+1（至少 25 帧，上游的最小长度），输出帧数和输入完全一样。
- 只做一步推理（cfg 1.0），与上游示例相同。

### 自检

```bash
cd ~/comfyui/custom_nodes/ComfyUI-FlashVSR-Linux/tests
~/comfyui/.venv/bin/python test_units.py
~/comfyui/.venv/bin/python test_cpu_pipeline.py          # CPU 上要几分钟
~/comfyui/.venv/bin/python check_gpu.py                  # 注意力 kernel 对比参考实现
~/comfyui/.venv/bin/python check_gpu.py --model-dir ~/comfyui/models/FlashVSR-v1.1 --mode tiny
```

`check_gpu.py` 会打印每个后端和参考实现的余弦相似度（应 > 0.99）、耗时，以及用真实权重放大一段合成视频的速度和显存峰值。

### 目录结构

```text
__init__.py, nodes.py     ComfyUI 节点
runtime/attention.py      稀疏注意力分发（SpargeAttn / Block-Sparse-Attention / SDPA）
runtime/budget.py         运动感知的分块预算
runtime/tiling.py         空间分块与融合权重
runtime/runner.py         模型加载与完整推理流程（不依赖 ComfyUI）
runtime/video.py          ffmpeg 视频写入
flashvsr/                 上游 FlashVSR 推理代码（见 NOTICE）
web/flashvsr_linux.js     Video Combine 节点内预览
tests/                    CPU 测试与 GPU 自检
```

### 许可证

Apache License 2.0，见 `LICENSE` 和 `NOTICE`。

<a id="english"></a>
## English

Open-source (Apache-2.0) ComfyUI nodes for running [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) video super-resolution on Linux.

### Background

[TE-Speed-FlashVSR](https://github.com/tl2012tl/TE-Speed-FlashVSR) is a good FlashVSR node pack, but its core is a closed-source
Windows-only binary (`nodes.pyd`, `te_runtime/*.pyd`) that cannot be loaded on Linux. This project follows the node design and
parameters described in its public README and reimplements them on top of the open-source upstream FlashVSR code:

- It does **not** use any code from the TE-Speed-FlashVSR binaries. Dynamic budgeting, tiling and video writing are independent
  implementations, so quality and speed will not exactly match the TE version.
- It is not affiliated with or endorsed by the authors of TE-Speed-FlashVSR or FlashVSR.
- The model inference code comes from upstream FlashVSR (Apache-2.0). Changes are listed in `NOTICE` and
  `patches/flashvsr-upstream-changes.diff`.

### Test status (read first)

| Item | Status |
|---|---|
| CPU end-to-end (scaled-down model with random weights; tiny / tiny-long / full, v1.0 / v1.1, tiling, staged VRAM, dynamic budget) | ✅ `tests/test_cpu_pipeline.py` 8/8 |
| Unit tests (block attention vs. brute-force reference, pixel-exact tile stitching, filename variables, ffmpeg encoding + audio, nodes) | ✅ `tests/test_units.py` 20/20 |
| sparse_sage2 / block_sparse_attn on GPU match the reference numerically | ⏳ pending: `tests/check_gpu.py` |
| Quality and speed with the real weights | ⏳ pending (the development environment had no GPU and could not download the weights) |

### Installation

1. Clone into ComfyUI's `custom_nodes`:

   ```bash
   cd ~/comfyui/custom_nodes
   git clone https://github.com/caoddgao/ComfyUI-FlashVSR-Linux.git
   ```

2. Install at least one sparse attention wheel. It must match your torch / CUDA / GPU.
   [Releases](https://github.com/caoddgao/ComfyUI-FlashVSR-Linux/releases) has wheels built for torch 2.14.1+cu130 and
   RTX 50 series (sm_120); the build and acceptance log is in
   [BUILD_REPORT](https://github.com/caoddgao/TE-Speed-FlashVSR/blob/claude/adoring-ramanujan-ytnixo/linux-build/BUILD_REPORT.md).
   Other GPUs / torch versions need to build from source
   ([SpargeAttn](https://github.com/woct0rdho/SpargeAttn),
   [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)).

   ```bash
   ~/comfyui/.venv/bin/python -m pip install --no-deps spas_sage_attn-0.1.0+cu130torch2.14.1-cp310-abi3-linux_x86_64.whl
   ~/comfyui/.venv/bin/python -m pip install --no-deps block_sparse_attn-0.0.2+cu130torch2.14.1-cp312-cp312-linux_x86_64.whl  # optional
   ```

3. `ffmpeg` is required (`sudo apt install ffmpeg`). `einops`, `safetensors` and `tqdm` are usually already installed with ComfyUI.

4. Put the models in `ComfyUI/models/FlashVSR-v1.1/` (or `FlashVSR/` for v1.0):

   ```text
   diffusion_pytorch_model_streaming_dmd.safetensors
   LQ_proj_in.ckpt
   TCDecoder.ckpt          # tiny / tiny-long
   Wan2.1_VAE.pth          # full
   posi_prompt.pth         # optional; the bundled copy from upstream is used if missing
   ```

   A folder whose name contains `1.1` is loaded as v1.1 (causal LQ projection), otherwise as v1.0.

   The models do not have to be under `ComfyUI/models/`; the nodes also search:
   - folders whose name contains `flashvsr` in the parent directory of each model path from `extra_model_paths.yaml`
     (usually the `base_path`);
   - a `flashvsr:` entry in `extra_model_paths.yaml`, pointing either at a model folder or at a directory containing several
     model folders, for example:

     ```yaml
     my_models:
         base_path: /mnt/data/AI_Hub/image_models
         checkpoints: checkpoints
         flashvsr: FlashVSR-v1.1
     ```

If TE-Speed-FlashVSR was installed before, it cannot load on Linux anyway and can be moved out of `custom_nodes`. The node names
differ, so the two do not conflict.

### Nodes

| Node | Purpose |
|---|---|
| **FlashVSR Model (Linux)** | Loads the model: `model` (model folder), `mode` (tiny / tiny-long / full), `precision` (bf16 / fp16), `device` |
| **FlashVSR Settings (Linux)** | Optional inference settings, connected to Upscale's `settings` |
| **FlashVSR Upscale (Linux)** | Takes an `IMAGE` frame batch and outputs the upscaled `IMAGE`: `scale` (4 / 2), `color_fix`, `seed` |
| **FlashVSR Video Combine (Linux)** | Writes mp4 with ffmpeg and previews it in the node; `codec` auto prefers `h264_nvenc` and falls back to `libx264`; accepts `audio` |

Typical wiring: Load Video → FlashVSR Upscale (Linux) → FlashVSR Video Combine (Linux), with the Model node connected to Upscale.

#### Settings

- `quality_profile`: `detail` (sparse 2.0 / local 9), `balanced` (2.0 / 11, default), `throughput` (1.5 / 11).
  These are upstream's recommended values: sparse_ratio 1.5 is faster, 2.0 is more stable; local_range 9 is sharper, 11 is more stable.
- `intensity`: strength of the motion-aware dynamic budget; 0 = fixed parameters throughout, like upstream. The algorithm is in
  `runtime/budget.py`: it estimates per-chunk motion (0–1) from frame differences of the low-res input, keeps fewer top-k windows
  for static chunks (down to −25%) with local 9, and more for moving chunks (up to +25%) with local 11; when
  `intensity × motion > 0.5` it also keeps one fewer history KV block (at the default intensity of 0.5 KV is unchanged). This is
  this project's own heuristic and needs real-world testing; set it to 0 if you don't like it.
- `spatial_strategy`: `auto` (full frame when output is at most about 1920×1088 and VRAM ≥ 20GB, otherwise tiles), `full_frame`,
  `adaptive_tiles`.
- `memory_policy`: `resident` (model stays in VRAM), `staged` (moves the DiT back to RAM before decoding; full mode uses tiled VAE
  decoding), `auto` (staged when free VRAM < 24GB or when tiling).
- `attention_budget` / `kv_retention`: set sparse_ratio / kv_ratio directly; -1 uses the profile value.
- `local_radius`: `auto` / `9` / `11`; when set to 9 or 11 the dynamic budget no longer changes it.
- `max_tile_edge` / `blend_overlap`: tile edge length and overlap (in input pixels). Tile output size is rounded to a multiple of 128.
- `preprocess_batch`: number of frames per batch for bicubic upscaling.
- `attention_backend`: `sparse_sage2` (default, SpargeAttn), `block_sparse_attn` (upstream's original kernel),
  `auto` (block_sparse_attn for full frames, sparse_sage2 for tiles), `sdpa` (very slow dense reference, for comparison only).
  If the selected library is unavailable it falls back to the other one and prints the reason to the console.

#### Behavior matching upstream

- Input is bicubic-upscaled to the target size, padded to a multiple of 128, and the result is cropped back to the original size
  (upstream center-crops and loses the edges; this project does not).
- Frame count is padded with the last frame to 8k+1 (at least 25 frames, upstream's minimum); output has exactly as many frames as
  the input.
- Only one inference step is run (cfg 1.0), the same as upstream's example.

### Self-check

```bash
cd ~/comfyui/custom_nodes/ComfyUI-FlashVSR-Linux/tests
~/comfyui/.venv/bin/python test_units.py
~/comfyui/.venv/bin/python test_cpu_pipeline.py          # takes a few minutes on CPU
~/comfyui/.venv/bin/python check_gpu.py                  # compares attention kernels with the reference
~/comfyui/.venv/bin/python check_gpu.py --model-dir ~/comfyui/models/FlashVSR-v1.1 --mode tiny
```

`check_gpu.py` prints each backend's cosine similarity against the reference (should be > 0.99) and its timing, plus the speed
and peak VRAM of upscaling a synthetic video with the real weights.

### Layout

```text
__init__.py, nodes.py     ComfyUI nodes
runtime/attention.py      sparse attention dispatch (SpargeAttn / Block-Sparse-Attention / SDPA)
runtime/budget.py         motion-aware per-chunk budget
runtime/tiling.py         spatial tiling and blend weights
runtime/runner.py         model loading and the full inference pipeline (no ComfyUI dependency)
runtime/video.py          ffmpeg video writer
flashvsr/                 upstream FlashVSR inference code (see NOTICE)
web/flashvsr_linux.js     in-node preview for Video Combine
tests/                    CPU tests and GPU self-check
```

### License

Apache License 2.0, see `LICENSE` and `NOTICE`.
