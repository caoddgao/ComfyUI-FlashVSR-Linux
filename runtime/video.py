"""Write ComfyUI IMAGE batches to a video file with ffmpeg (no ComfyUI imports)."""

import datetime
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import wave

import torch

_ENCODER_OK = {}


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found: install it (apt install ffmpeg) or pip install imageio-ffmpeg")
    return exe


def encoder_works(exe, codec):
    """Try a tiny encode once per codec; NVENC can be compiled in but unusable (no driver/GPU)."""
    if codec in _ENCODER_OK:
        return _ENCODER_OK[codec]
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
           "-frames:v", "2", "-c:v", codec, "-f", "null", "-"]
    try:
        ok = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except Exception:
        ok = False
    _ENCODER_OK[codec] = ok
    return ok


def pick_codec(exe, codec):
    if codec != "auto":
        return codec
    for candidate in ("h264_nvenc", "libx264"):
        if encoder_works(exe, candidate):
            return candidate
    return "mpeg4"


def codec_args(codec, quality):
    """quality: 0 (best) .. 51 (worst), like x264 CRF."""
    if codec.endswith("_nvenc"):
        return ["-c:v", codec, "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
    if codec in ("libx264", "libx265"):
        return ["-c:v", codec, "-preset", "medium", "-crf", str(quality)]
    return ["-c:v", codec, "-q:v", str(max(1, min(31, quality // 2)))]


_DATE_TOKENS = [("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"), ("HH", "%H"), ("mm", "%M"),
                ("ss", "%S")]


def expand_prefix(prefix, width, height, now=None):
    """Expand %date:yyyyMMdd_HHmmss%, %year% ... %second%, %width%, %height% in a filename prefix."""
    now = now or datetime.datetime.now()

    def date_repl(match):
        fmt = match.group(1)
        out = ""
        i = 0
        while i < len(fmt):
            if fmt.startswith("fff", i):
                out += f"{now.microsecond // 1000:03d}"
                i += 3
                continue
            for token, code in _DATE_TOKENS:
                if fmt.startswith(token, i):
                    out += now.strftime(code)
                    i += len(token)
                    break
            else:
                out += fmt[i]
                i += 1
        return out

    prefix = re.sub(r"%date:([^%]+)%", date_repl, prefix)
    simple = {"year": "%Y", "month": "%m", "day": "%d", "hour": "%H", "minute": "%M", "second": "%S"}
    for key, code in simple.items():
        prefix = prefix.replace(f"%{key}%", now.strftime(code))
    return prefix.replace("%width%", str(width)).replace("%height%", str(height))


def next_path(directory, prefix, ext="mp4"):
    """<directory>/<prefix>_00001.<ext>, picking the first free counter. prefix may contain sub folders."""
    sub, base = os.path.split(prefix.replace("\\", "/"))
    folder = os.path.abspath(os.path.join(directory, sub))
    if os.path.commonpath([folder, os.path.abspath(directory)]) != os.path.abspath(directory):
        raise ValueError("filename_prefix must stay inside the output folder")
    os.makedirs(folder, exist_ok=True)
    pattern = re.compile(re.escape(base) + r"_(\d{5,})\." + re.escape(ext) + "$")
    counter = 1
    for name in os.listdir(folder):
        m = pattern.match(name)
        if m:
            counter = max(counter, int(m.group(1)) + 1)
    filename = f"{base}_{counter:05d}.{ext}"
    return os.path.join(folder, filename), sub, filename


def write_wav(path, audio):
    """ComfyUI AUDIO {"waveform": [B, C, S] float, "sample_rate": int} -> 16-bit PCM wav."""
    wav = audio["waveform"][0].float().clamp(-1, 1)
    channels = wav.shape[0]
    pcm = (wav.t().contiguous() * 32767.0).round().to(torch.int16).numpy().tobytes()
    with wave.open(path, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(int(audio["sample_rate"]))
        f.writeframes(pcm)


@torch.no_grad()
def encode(images, path, fps, codec="auto", quality=19, audio=None, workers=3, pix_fmt="yuv420p"):
    """Encode [N, H, W, 3] float 0..1 frames to ``path``. Returns the codec used.

    Conversion to uint8 runs in ``workers`` threads ahead of the ffmpeg writer.
    """
    exe = ffmpeg_exe()
    codec = pick_codec(exe, codec)
    n, h, w, _ = images.shape
    if pix_fmt == "yuv420p" and (h % 2 or w % 2):
        images = images[:, : h - h % 2, : w - w % 2]
        n, h, w, _ = images.shape

    tmpdir = tempfile.mkdtemp(prefix="flashvsr_audio_")
    try:
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
        if audio is not None:
            wav_path = os.path.join(tmpdir, "audio.wav")
            write_wav(wav_path, audio)
            # apad + -shortest: the video keeps every frame, audio is padded with silence or cut to fit
            cmd += ["-i", wav_path, "-map", "0:v", "-map", "1:a", "-af", "apad", "-c:a", "aac", "-b:a", "192k",
                    "-shortest"]
        cmd += codec_args(codec, quality) + ["-pix_fmt", pix_fmt, "-movflags", "+faststart", path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        chunk = 8
        jobs = queue.Queue()
        results = {}
        cond = threading.Condition()

        def convert():
            while True:
                start = jobs.get()
                if start is None:
                    return
                data = images[start:start + chunk].mul(255).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy().tobytes()
                with cond:
                    results[start] = data
                    cond.notify_all()

        threads = [threading.Thread(target=convert, daemon=True) for _ in range(max(1, workers))]
        for t in threads:
            t.start()
        starts = list(range(0, n, chunk))
        ahead = max(1, workers) * 2
        for s in starts[:ahead]:
            jobs.put(s)
        try:
            for i, s in enumerate(starts):
                with cond:
                    while s not in results:
                        cond.wait()
                    data = results.pop(s)
                if i + ahead < len(starts):
                    jobs.put(starts[i + ahead])
                proc.stdin.write(data)
        except BrokenPipeError:
            pass  # ffmpeg exited early; its stderr is reported below
        finally:
            for _ in threads:
                jobs.put(None)
            try:
                proc.stdin.close()
            except Exception:
                pass
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed ({codec}): {err.strip()}")
        return codec
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
