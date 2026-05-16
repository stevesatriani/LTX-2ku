"""
LTX-2 Web UI
Gradio-based frontend for all LTX-2 pipelines, designed for SimplePod.

Supported pipelines:
  1.  TI2VidTwoStagesPipeline      – Production quality text/image-to-video (recommended)
  2.  TI2VidTwoStagesHQPipeline    – Same two-stage flow, res_2s sampler, fewer steps
  3.  TI2VidOneStagePipeline       – Single-stage, quick prototyping
  4.  DistilledPipeline            – Fastest inference, 8 predefined sigma steps
  5.  ICLoraPipeline               – Video-to-video with IC-LoRA (distilled model)
  6.  KeyframeInterpolationPipeline – Interpolate between keyframe images
  7.  A2VidPipelineTwoStage        – Audio-conditioned video generation
  8.  RetakePipeline               – Regenerate a time region of an existing video
  9.  HDRICLoraPipeline            – HDR video-to-video (linear float / EXR output)
  10. LipDubPipeline               – Lip dubbing with audio reference conditioning

Usage:
    python app.py
    python app.py --port 7860 --share

Environment variables (optional, pre-fill UI paths):
    LTX_CHECKPOINT_PATH      – Main model checkpoint (.safetensors)
    LTX_DISTILLED_CHECKPOINT – Distilled checkpoint (.safetensors)
    LTX_GEMMA_ROOT           – Gemma text encoder directory
    LTX_UPSAMPLER_PATH       – Spatial upsampler (.safetensors)
    LTX_DISTILLED_LORA_PATH  – Distilled LoRA (.safetensors)
    LTX_OUTPUT_DIR           – Output directory (default: ./outputs)
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

# ─── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.environ.get("LTX_OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Default paths from environment ──────────────────────────────────────────
ENV = {
    "checkpoint": os.environ.get("LTX_CHECKPOINT_PATH", ""),
    "distilled_ckpt": os.environ.get("LTX_DISTILLED_CHECKPOINT", ""),
    "gemma": os.environ.get("LTX_GEMMA_ROOT", ""),
    "upsampler": os.environ.get("LTX_UPSAMPLER_PATH", ""),
    "distilled_lora": os.environ.get("LTX_DISTILLED_LORA_PATH", ""),
}

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

# ─── CSS (lovable.dev aesthetic) ──────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#ltx-header {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 55%, #ec4899 100%);
    border-radius: 20px;
    padding: 26px 36px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 12px 40px rgba(99,102,241,.35);
}
#ltx-header h1 { font-size:2.1rem; font-weight:800; margin:0 0 4px; letter-spacing:-.5px; }
#ltx-header .sub { font-size:.97rem; opacity:.88; margin:0; }
#ltx-header .badges span {
    display:inline-block; background:rgba(255,255,255,.22); border-radius:20px;
    padding:3px 12px; font-size:.75rem; font-weight:600; margin:8px 4px 0;
}
.tab-nav button { border-radius:10px !important; font-weight:600 !important; }
.tab-nav button.selected {
    background: linear-gradient(135deg,#6366f1,#a855f7) !important;
    color:white !important; box-shadow:0 4px 12px rgba(99,102,241,.35) !important;
}
.generate-btn {
    background: linear-gradient(135deg,#6366f1 0%,#a855f7 100%) !important;
    border:none !important; border-radius:12px !important;
    padding:14px 28px !important; font-size:1.02rem !important;
    font-weight:700 !important; color:white !important;
    box-shadow:0 4px 15px rgba(99,102,241,.35) !important;
    transition:all .25s ease !important; min-height:52px !important;
}
.generate-btn:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(99,102,241,.55) !important;
}
.pipe-info {
    background:linear-gradient(135deg,#ede9fe,#fce7f3);
    border:1px solid #c4b5fd; border-radius:10px;
    padding:10px 14px; font-size:.84rem; color:#5b21b6; margin-bottom:12px;
}
#log-out textarea {
    font-family:'JetBrains Mono','Fira Code','Courier New',monospace !important;
    font-size:.77rem !important; background:#0f172a !important;
    color:#7dd3fc !important; border:1px solid #1e3a5f !important;
    border-radius:12px !important; line-height:1.6 !important;
}
#video-out video { border-radius:14px; box-shadow:0 4px 20px rgba(0,0,0,.15); }
label { font-weight:500 !important; font-size:.88rem !important; color:#374151 !important; }
input[type="range"] { accent-color:#6366f1 !important; }
.accordion .label-wrap {
    background:#f8fafc !important; border:1px solid #e2e8f0 !important;
    border-radius:12px !important; font-weight:600 !important;
    color:#374151 !important; padding:12px 16px !important;
}
"""

# ─── Shared helpers ────────────────────────────────────────────────────────────

def _out(prefix: str = "video") -> str:
    """Return a unique timestamped output path inside OUTPUT_DIR."""
    return str(OUTPUT_DIR / f"{prefix}_{int(time.time()*1000)}.mp4")


def _ok(path: str, label: str = "Path") -> tuple[bool, str]:
    """Validate that a path is provided and exists."""
    if not path or not path.strip():
        return False, f"⚠️  {label} is empty."
    if not Path(path.strip()).exists():
        return False, f"⚠️  {label} not found:\n  {path.strip()}"
    return True, "ok"


def _stream(cmd: list[str], output_path: str):
    """
    Run a subprocess, stream stdout+stderr live to Gradio, then yield final video.
    Yields (video_or_None, log_text) tuples for use as a Gradio generator output.
    Sets PYTORCH_CUDA_ALLOC_CONF for LTX-2 stability.
    """
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    lines: list[str] = []
    for line in iter(proc.stdout.readline, ""):
        lines.append(line)
        yield None, "".join(lines)
    proc.wait()
    log = "".join(lines)
    if proc.returncode == 0 and Path(output_path).exists():
        yield output_path, log + "\n\n✅  Done!"
    elif proc.returncode == 0:
        yield None, log + f"\n⚠️  Finished but output not found:\n  {output_path}"
    else:
        yield None, log + f"\n\n❌  Failed  (exit code {proc.returncode})"


def _stream_dir_out(cmd: list[str], output_dir: str):
    """
    Like _stream, but scans a directory for the newest MP4 after completion.
    Used by HDRICLoraPipeline which writes to a directory, not a single file.
    """
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    lines: list[str] = []
    for line in iter(proc.stdout.readline, ""):
        lines.append(line)
        yield None, "".join(lines)
    proc.wait()
    log = "".join(lines)
    if proc.returncode != 0:
        yield None, log + f"\n\n❌  Failed  (exit code {proc.returncode})"
        return
    # Find newest MP4 in output directory
    mp4s = sorted(glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
    if mp4s:
        yield mp4s[-1], log + "\n\n✅  Done!  (HDR EXR frames also written to output dir)"
    else:
        yield None, log + f"\n\n✅  Done!  No MP4 preview found (check output dir: {output_dir})"


# ─── Pipeline runner functions ─────────────────────────────────────────────────

def run_ti2vid(
    ckpt, gemma, upsamp, dlora, dlora_str,
    prompt, neg_prompt, height, width, n_frames, fps, steps, seed,
    quant, offload, enhance,
    img, img_fidx, img_str,
):
    """TI2VidTwoStagesPipeline – two-stage recommended pipeline."""
    for lbl, p in [("Checkpoint", ckpt), ("Gemma", gemma),
                   ("Upsampler", upsamp), ("Distilled LoRA", dlora)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    out = _out("ti2vid")
    cmd = [sys.executable, "-m", "ltx_pipelines.ti2vid_two_stages",
           "--checkpoint-path", ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--spatial-upsampler-path", upsamp.strip(),
           "--distilled-lora", dlora.strip(), str(round(dlora_str, 3)),
           "--prompt", prompt, "--negative-prompt", neg_prompt,
           "--height", str(int(height)), "--width", str(int(width)),
           "--num-frames", str(int(n_frames)), "--frame-rate", str(float(fps)),
           "--num-inference-steps", str(int(steps)), "--seed", str(int(seed)),
           "--output-path", out]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    if enhance: cmd += ["--enhance-prompt"]
    if img: cmd += ["--image", img, str(int(img_fidx)), str(round(img_str, 3))]
    yield None, "🚀  TI2Vid Two-Stage starting...\n\n"
    yield from _stream(cmd, out)


def run_ti2vid_hq(
    ckpt, gemma, upsamp, dlora, dlora_str_s1, dlora_str_s2,
    prompt, neg_prompt, height, width, n_frames, fps, steps, seed,
    quant, offload, enhance,
    img, img_fidx, img_str,
):
    """TI2VidTwoStagesHQPipeline – res_2s sampler, higher quality, fewer steps."""
    for lbl, p in [("Checkpoint", ckpt), ("Gemma", gemma),
                   ("Upsampler", upsamp), ("Distilled LoRA", dlora)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    out = _out("ti2vid_hq")
    cmd = [sys.executable, "-m", "ltx_pipelines.ti2vid_two_stages_hq",
           "--checkpoint-path", ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--spatial-upsampler-path", upsamp.strip(),
           "--distilled-lora", dlora.strip(), str(round(max(dlora_str_s1, dlora_str_s2), 3)),
           "--distilled-lora-strength-stage-1", str(round(dlora_str_s1, 3)),
           "--distilled-lora-strength-stage-2", str(round(dlora_str_s2, 3)),
           "--prompt", prompt, "--negative-prompt", neg_prompt,
           "--height", str(int(height)), "--width", str(int(width)),
           "--num-frames", str(int(n_frames)), "--frame-rate", str(float(fps)),
           "--num-inference-steps", str(int(steps)), "--seed", str(int(seed)),
           "--output-path", out]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    if enhance: cmd += ["--enhance-prompt"]
    if img: cmd += ["--image", img, str(int(img_fidx)), str(round(img_str, 3))]
    yield None, "🏆  TI2Vid HQ (res_2s sampler) starting...\n\n"
    yield from _stream(cmd, out)


def run_one_stage(
    ckpt, gemma,
    prompt, neg_prompt, height, width, n_frames, fps, steps, seed,
    quant, offload, enhance,
    img, img_fidx, img_str,
):
    """TI2VidOneStagePipeline – single-stage, no upsampling, quick prototyping."""
    for lbl, p in [("Checkpoint", ckpt), ("Gemma", gemma)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    out = _out("one_stage")
    cmd = [sys.executable, "-m", "ltx_pipelines.ti2vid_one_stage",
           "--checkpoint-path", ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--prompt", prompt, "--negative-prompt", neg_prompt,
           "--height", str(int(height)), "--width", str(int(width)),
           "--num-frames", str(int(n_frames)), "--frame-rate", str(float(fps)),
           "--num-inference-steps", str(int(steps)), "--seed", str(int(seed)),
           "--output-path", out]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    if enhance: cmd += ["--enhance-prompt"]
    if img: cmd += ["--image", img, str(int(img_fidx)), str(round(img_str, 3))]
    yield None, "🔬  One-Stage starting (no upsampling)...\n\n"
    yield from _stream(cmd, out)


def run_distilled(
    dist_ckpt, gemma, upsamp,
    prompt, height, width, n_frames, fps, seed,
    quant, offload, enhance,
    img, img_fidx, img_str,
):
    """DistilledPipeline – fastest, 8 predefined sigma steps."""
    for lbl, p in [("Distilled checkpoint", dist_ckpt),
                   ("Gemma", gemma), ("Upsampler", upsamp)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    out = _out("distilled")
    cmd = [sys.executable, "-m", "ltx_pipelines.distilled",
           "--distilled-checkpoint-path", dist_ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--spatial-upsampler-path", upsamp.strip(),
           "--prompt", prompt,
           "--height", str(int(height)), "--width", str(int(width)),
           "--num-frames", str(int(n_frames)), "--frame-rate", str(float(fps)),
           "--seed", str(int(seed)), "--output-path", out]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    if enhance: cmd += ["--enhance-prompt"]
    if img: cmd += ["--image", img, str(int(img_fidx)), str(round(img_str, 3))]
    yield None, "⚡  Distilled (fastest mode) starting...\n\n"
    yield from _stream(cmd, out)


def run_ic_lora(
    dist_ckpt, gemma, upsamp, ic_lora, ic_lora_str,
    prompt, height, width, n_frames, fps, seed,
    quant, offload, enhance,
    vid_cond, vid_cond_str,
    vid_cond2, vid_cond2_str,
    img, img_fidx, img_str,
    skip_stage2,
):
    """ICLoraPipeline – video-to-video with IC-LoRA (uses distilled checkpoint)."""
    for lbl, p in [("Distilled checkpoint", dist_ckpt),
                   ("Gemma", gemma), ("Upsampler", upsamp)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    if not vid_cond or not vid_cond.strip():
        yield None, "⚠️  At least one video conditioning path is required."; return
    ok, msg = _ok(vid_cond, "Video conditioning")
    if not ok:
        yield None, msg; return
    out = _out("ic_lora")
    cmd = [sys.executable, "-m", "ltx_pipelines.ic_lora",
           "--distilled-checkpoint-path", dist_ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--spatial-upsampler-path", upsamp.strip(),
           "--prompt", prompt,
           "--height", str(int(height)), "--width", str(int(width)),
           "--num-frames", str(int(n_frames)), "--frame-rate", str(float(fps)),
           "--seed", str(int(seed)), "--output-path", out,
           "--video-conditioning", vid_cond.strip(), str(round(vid_cond_str, 3))]
    if vid_cond2 and vid_cond2.strip() and Path(vid_cond2.strip()).exists():
        cmd += ["--video-conditioning", vid_cond2.strip(), str(round(vid_cond2_str, 3))]
    if ic_lora and ic_lora.strip() and Path(ic_lora.strip()).exists():
        cmd += ["--lora", ic_lora.strip(), str(round(ic_lora_str, 3))]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    if enhance: cmd += ["--enhance-prompt"]
    if img: cmd += ["--image", img, str(int(img_fidx)), str(round(img_str, 3))]
    if skip_stage2: cmd += ["--skip-stage-2"]
    yield None, "🎭  IC-LoRA (video-to-video) starting...\n\n"
    yield from _stream(cmd, out)


def run_keyframe(
    ckpt, gemma, upsamp, dlora, dlora_str,
    prompt, neg_prompt, height, width, n_frames, fps, steps, seed,
    quant, offload, enhance,
    kf1_path, kf1_fidx, kf1_str,
    kf2_path, kf2_fidx, kf2_str,
    kf3_path, kf3_fidx, kf3_str,
    kf4_path, kf4_fidx, kf4_str,
):
    """KeyframeInterpolationPipeline – interpolate smoothly between keyframe images."""
    for lbl, p in [("Checkpoint", ckpt), ("Gemma", gemma),
                   ("Upsampler", upsamp), ("Distilled LoRA", dlora)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    # At least one keyframe required
    keyframes = [
        (kf1_path, kf1_fidx, kf1_str), (kf2_path, kf2_fidx, kf2_str),
        (kf3_path, kf3_fidx, kf3_str), (kf4_path, kf4_fidx, kf4_str),
    ]
    valid_kf = [(p, f, s) for p, f, s in keyframes if p and p.strip() and Path(p.strip()).exists()]
    if not valid_kf:
        yield None, "⚠️  At least one valid keyframe image path is required."; return
    out = _out("keyframe")
    cmd = [sys.executable, "-m", "ltx_pipelines.keyframe_interpolation",
           "--checkpoint-path", ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--spatial-upsampler-path", upsamp.strip(),
           "--distilled-lora", dlora.strip(), str(round(dlora_str, 3)),
           "--prompt", prompt, "--negative-prompt", neg_prompt,
           "--height", str(int(height)), "--width", str(int(width)),
           "--num-frames", str(int(n_frames)), "--frame-rate", str(float(fps)),
           "--num-inference-steps", str(int(steps)), "--seed", str(int(seed)),
           "--output-path", out]
    for kf_path, kf_fidx, kf_str in valid_kf:
        cmd += ["--image", kf_path.strip(), str(int(kf_fidx)), str(round(kf_str, 3))]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    if enhance: cmd += ["--enhance-prompt"]
    yield None, f"🖼️  Keyframe Interpolation starting ({len(valid_kf)} keyframes)...\n\n"
    yield from _stream(cmd, out)


def run_a2vid(
    ckpt, gemma, upsamp, dlora, dlora_str,
    prompt, neg_prompt, height, width, n_frames, fps, steps, seed,
    quant, offload, enhance,
    audio_path, audio_start, audio_max_dur,
    img, img_fidx, img_str,
):
    """A2VidPipelineTwoStage – audio-conditioned two-stage video generation."""
    for lbl, p in [("Checkpoint", ckpt), ("Gemma", gemma),
                   ("Upsampler", upsamp), ("Distilled LoRA", dlora)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    if not audio_path or not audio_path.strip():
        yield None, "⚠️  Audio path is required."; return
    ok, msg = _ok(audio_path, "Audio")
    if not ok:
        yield None, msg; return
    out = _out("a2vid")
    cmd = [sys.executable, "-m", "ltx_pipelines.a2vid_two_stage",
           "--checkpoint-path", ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--spatial-upsampler-path", upsamp.strip(),
           "--distilled-lora", dlora.strip(), str(round(dlora_str, 3)),
           "--prompt", prompt, "--negative-prompt", neg_prompt,
           "--height", str(int(height)), "--width", str(int(width)),
           "--num-frames", str(int(n_frames)), "--frame-rate", str(float(fps)),
           "--num-inference-steps", str(int(steps)), "--seed", str(int(seed)),
           "--audio-path", audio_path.strip(), "--output-path", out]
    if audio_start > 0: cmd += ["--audio-start-time", str(float(audio_start))]
    if audio_max_dur > 0: cmd += ["--audio-max-duration", str(float(audio_max_dur))]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    if enhance: cmd += ["--enhance-prompt"]
    if img: cmd += ["--image", img, str(int(img_fidx)), str(round(img_str, 3))]
    yield None, "🎵  Audio-to-Video starting...\n\n"
    yield from _stream(cmd, out)


def run_retake(
    dist_ckpt, gemma,
    prompt, neg_prompt, steps, seed, quant, offload,
    vid_path, start_t, end_t,
):
    """RetakePipeline – regenerate a specific time window of an existing video."""
    for lbl, p in [("Distilled checkpoint", dist_ckpt), ("Gemma", gemma)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    if not vid_path or not vid_path.strip():
        yield None, "⚠️  Source video path is required."; return
    ok, msg = _ok(vid_path, "Source video")
    if not ok:
        yield None, msg; return
    if start_t >= end_t:
        yield None, "⚠️  Start time must be less than end time."; return
    out = _out("retake")
    cmd = [sys.executable, "-m", "ltx_pipelines.retake",
           "--distilled-checkpoint-path", dist_ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--prompt", prompt, "--negative-prompt", neg_prompt,
           "--num-inference-steps", str(int(steps)), "--seed", str(int(seed)),
           "--video-path", vid_path.strip(),
           "--start-time", str(float(start_t)), "--end-time", str(float(end_t)),
           "--output-path", out]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    yield None, f"✂️  Retake [{start_t}s → {end_t}s] starting...\n\n"
    yield from _stream(cmd, out)


def run_hdr_ic_lora(
    dist_ckpt, upsamp, hdr_lora, text_emb,
    input_path, n_frames, seed, offload,
    skip_mp4, exr_half, high_quality, spatial_tile,
):
    """
    HDRICLoraPipeline – video-to-video with HDR output (EXR linear frames).
    Text embeddings must be pre-computed; no Gemma encoder needed at runtime.
    """
    for lbl, p in [("Distilled checkpoint", dist_ckpt),
                   ("Upsampler", upsamp), ("HDR LoRA", hdr_lora),
                   ("Text embeddings", text_emb)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    if not input_path or not input_path.strip():
        yield None, "⚠️  Input video/directory path is required."; return
    ok, msg = _ok(input_path, "Input video/dir")
    if not ok:
        yield None, msg; return
    hdr_out_dir = str(OUTPUT_DIR / f"hdr_{int(time.time()*1000)}")
    cmd = [sys.executable, "-m", "ltx_pipelines.hdr_ic_lora",
           "--distilled-checkpoint-path", dist_ckpt.strip(),
           "--spatial-upsampler-path", upsamp.strip(),
           "--hdr-lora", hdr_lora.strip(),
           "--text-embeddings", text_emb.strip(),
           "--input", input_path.strip(),
           "--output-dir", hdr_out_dir,
           "--num-frames", str(int(n_frames)),
           "--seed", str(int(seed)),
           "--spatial-tile", str(int(spatial_tile))]
    if offload != "none": cmd += ["--offload", offload]
    if skip_mp4: cmd += ["--skip-mp4"]
    if exr_half: cmd += ["--exr-half"]
    if high_quality: cmd += ["--high-quality"]
    yield None, f"🌟  HDR IC-LoRA starting...\nOutput dir: {hdr_out_dir}\n\n"
    yield from _stream_dir_out(cmd, hdr_out_dir)


def run_lipdub(
    dist_ckpt, gemma, upsamp, ic_lora, ic_lora_str,
    prompt, height, width, seed, quant, offload,
    ref_video, ref_strength,
):
    """LipDubPipeline – lip dubbing with IC-LoRA + audio reference conditioning."""
    for lbl, p in [("Distilled checkpoint", dist_ckpt),
                   ("Gemma", gemma), ("Upsampler", upsamp)]:
        ok, msg = _ok(p, lbl)
        if not ok:
            yield None, msg; return
    if not ref_video or not ref_video.strip():
        yield None, "⚠️  Reference video is required."; return
    ok, msg = _ok(ref_video, "Reference video")
    if not ok:
        yield None, msg; return
    out = _out("lipdub")
    cmd = [sys.executable, "-m", "ltx_pipelines.lipdub",
           "--distilled-checkpoint-path", dist_ckpt.strip(),
           "--gemma-root", gemma.strip(),
           "--spatial-upsampler-path", upsamp.strip(),
           "--reference-video", ref_video.strip(),
           "--reference-strength", str(round(ref_strength, 3)),
           "--prompt", prompt,
           "--height", str(int(height)), "--width", str(int(width)),
           "--seed", str(int(seed)), "--output-path", out]
    if ic_lora and ic_lora.strip() and Path(ic_lora.strip()).exists():
        cmd += ["--lora", ic_lora.strip(), str(round(ic_lora_str, 3))]
    if quant != "none": cmd += ["--quantization", quant]
    if offload != "none": cmd += ["--offload", offload]
    yield None, "👄  LipDub starting...\n\n"
    yield from _stream(cmd, out)


# ─── Reusable UI component builders ───────────────────────────────────────────

def _paths_two_stage():
    """Model path accordion for full-model two-stage pipelines (TI2Vid, A2Vid, Keyframe)."""
    with gr.Accordion("⚙️  Model Paths", open=True):
        with gr.Row():
            ckpt = gr.Textbox(label="Checkpoint (.safetensors)", value=ENV["checkpoint"],
                              placeholder="/models/ltx-2.3-22b-dev.safetensors")
            gemma = gr.Textbox(label="Gemma Root Directory", value=ENV["gemma"],
                               placeholder="/models/gemma-3-12b-it-qat-q4_0-unquantized")
        with gr.Row():
            upsamp = gr.Textbox(label="Spatial Upsampler (.safetensors)", value=ENV["upsampler"],
                                placeholder="/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
            dlora = gr.Textbox(label="Distilled LoRA (.safetensors)", value=ENV["distilled_lora"],
                               placeholder="/models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
            dlora_str = gr.Slider(label="Distilled LoRA Strength", minimum=0.0, maximum=1.5,
                                  step=0.05, value=0.8)
    return ckpt, gemma, upsamp, dlora, dlora_str


def _paths_distilled_only():
    """Model path accordion for distilled-only pipelines (Distilled, IC-LoRA, Retake, LipDub)."""
    with gr.Accordion("⚙️  Model Paths", open=True):
        with gr.Row():
            dist_ckpt = gr.Textbox(label="Distilled Checkpoint (.safetensors)",
                                   value=ENV["distilled_ckpt"],
                                   placeholder="/models/ltx-2.3-22b-distilled-1.1.safetensors")
            gemma = gr.Textbox(label="Gemma Root Directory", value=ENV["gemma"],
                               placeholder="/models/gemma")
    return dist_ckpt, gemma


def _prompt_row(lines=4):
    """Prompt + negative prompt side by side."""
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=lines,
                                placeholder="Describe what you want in the video, cinematographically...")
        with gr.Column(scale=2):
            neg = gr.Textbox(label="Negative Prompt", lines=lines, value=DEFAULT_NEGATIVE_PROMPT)
    return prompt, neg


def _vid_params(h=1024, w=1536):
    """Height, width, num_frames, fps sliders."""
    with gr.Row():
        height = gr.Slider(label="Height (px)", minimum=256, maximum=1920, step=64, value=h)
        width  = gr.Slider(label="Width (px)",  minimum=256, maximum=1920, step=64, value=w)
    with gr.Row():
        n_frames = gr.Slider(label="Frames (8k+1 format)",
                             minimum=9, maximum=257, step=8, value=121,
                             info="Valid: 9, 17, 25 … 121, 193, 257")
        fps = gr.Slider(label="Frame Rate (fps)", minimum=8.0, maximum=60.0, step=1.0, value=24.0)
    return height, width, n_frames, fps


def _adv(default_steps=30):
    """Advanced params accordion: steps, seed, quantization, offload, enhance."""
    with gr.Accordion("🔧  Advanced Parameters", open=False):
        with gr.Row():
            steps = gr.Slider(label="Inference Steps", minimum=4, maximum=60, step=1,
                              value=default_steps, info="More steps = better quality, slower")
            seed = gr.Number(label="Seed", value=42, precision=0)
        with gr.Row():
            quant = gr.Dropdown(label="Quantization",
                                choices=["none", "fp8-cast", "fp8-scaled-mm"], value="fp8-cast",
                                info="fp8-cast reduces VRAM ~50%; fp8-scaled-mm needs TRT-LLM")
            offload = gr.Dropdown(label="Weight Offload",
                                  choices=["none", "cpu", "disk"], value="none",
                                  info="Use cpu/disk if VRAM is insufficient")
        enhance = gr.Checkbox(label="✨ Auto-enhance prompt", value=False)
    return steps, seed, quant, offload, enhance


def _adv_no_steps():
    """Advanced params without steps (for distilled / LipDub pipelines)."""
    with gr.Accordion("🔧  Advanced Parameters", open=False):
        with gr.Row():
            seed = gr.Number(label="Seed", value=42, precision=0)
            quant = gr.Dropdown(label="Quantization",
                                choices=["none", "fp8-cast", "fp8-scaled-mm"], value="fp8-cast")
            offload = gr.Dropdown(label="Weight Offload",
                                  choices=["none", "cpu", "disk"], value="none")
        enhance = gr.Checkbox(label="✨ Auto-enhance prompt", value=False)
    return seed, quant, offload, enhance


def _img_cond():
    """Optional single image conditioning accordion."""
    with gr.Accordion("🖼️  Image Conditioning (optional)", open=False):
        gr.Markdown("_Condition on a reference image. Frame idx 0 = use as starting frame._")
        with gr.Row():
            img = gr.Textbox(label="Image Path", placeholder="/path/to/reference.jpg")
            fidx = gr.Number(label="Frame Index", value=0, precision=0)
            istr = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.05, value=1.0)
    return img, fidx, istr


def _output():
    """Output video player + live log panel."""
    with gr.Row():
        with gr.Column(scale=3):
            vid = gr.Video(label="Generated Video", elem_id="video-out", height=400)
        with gr.Column(scale=2):
            log = gr.Textbox(label="Live Log", lines=20, max_lines=20,
                             interactive=False, elem_id="log-out",
                             placeholder="Generation logs appear here in real-time...")
    return vid, log


def _gen_btn(label="🎬  Generate"):
    """Primary generate button with gradient style."""
    return gr.Button(label, variant="primary", elem_classes="generate-btn")


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """Build and return the complete Gradio Blocks application."""
    with gr.Blocks(
        title="LTX-2 Video Generation",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            radius_size=gr.themes.sizes.radius_lg,
        ),
        css=CUSTOM_CSS,
    ) as demo:

        gr.HTML("""
        <div id="ltx-header">
          <h1>🎬 LTX-2 Video Generation</h1>
          <p class="sub">
            Audio-video foundation model by Lightricks ·
            10 pipelines · text-to-video · image-to-video · audio-to-video · HDR · lip-dub
          </p>
          <div class="badges">
            <span>LTX-2.3 · 22B</span><span>Two-Stage</span>
            <span>Audio + Video</span><span>IC-LoRA</span><span>SimplePod Ready</span>
          </div>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # 1. TI2Vid (Recommended)
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("⭐ TI2Vid"):
                gr.HTML('<div class="pipe-info"><strong>TI2VidTwoStagesPipeline</strong>'
                        ' – Best quality production text/image-to-video. Stage 1 generates at'
                        ' half resolution with CFG guidance; Stage 2 upsamples 2× with distilled'
                        ' LoRA refinement. Synchronized audio output included.</div>')
                t1_ck, t1_gm, t1_up, t1_dl, t1_dls = _paths_two_stage()
                t1_pr, t1_np = _prompt_row()
                t1_h, t1_w, t1_nf, t1_fps = _vid_params()
                t1_st, t1_sd, t1_qu, t1_of, t1_en = _adv(30)
                t1_im, t1_if, t1_is = _img_cond()
                t1_btn = _gen_btn("🎬  Generate (TI2Vid)")
                t1_vid, t1_log = _output()
                t1_btn.click(run_ti2vid,
                    [t1_ck, t1_gm, t1_up, t1_dl, t1_dls,
                     t1_pr, t1_np, t1_h, t1_w, t1_nf, t1_fps, t1_st, t1_sd,
                     t1_qu, t1_of, t1_en, t1_im, t1_if, t1_is],
                    [t1_vid, t1_log])

            # ══════════════════════════════════════════════════════════════
            # 2. TI2Vid HQ
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🏆 TI2Vid HQ"):
                gr.HTML('<div class="pipe-info"><strong>TI2VidTwoStagesHQPipeline</strong>'
                        ' – Same two-stage flow but uses the res_2s second-order sampler.'
                        ' Achieves better quality with fewer inference steps (~15).'
                        ' Default output resolution: 1088×1920.</div>')
                t2_ck, t2_gm, t2_up, t2_dl, _ = _paths_two_stage()
                with gr.Accordion("⚙️  HQ LoRA Stage Strengths", open=False):
                    with gr.Row():
                        t2_dls1 = gr.Slider(label="Distilled LoRA Strength – Stage 1",
                                            minimum=0.0, maximum=1.0, step=0.05, value=0.25)
                        t2_dls2 = gr.Slider(label="Distilled LoRA Strength – Stage 2",
                                            minimum=0.0, maximum=1.0, step=0.05, value=0.5)
                t2_pr, t2_np = _prompt_row()
                t2_h, t2_w, t2_nf, t2_fps = _vid_params(h=1088, w=1920)
                t2_st, t2_sd, t2_qu, t2_of, t2_en = _adv(15)
                t2_im, t2_if, t2_is = _img_cond()
                t2_btn = _gen_btn("🏆  Generate (TI2Vid HQ)")
                t2_vid, t2_log = _output()
                t2_btn.click(run_ti2vid_hq,
                    [t2_ck, t2_gm, t2_up, t2_dl, t2_dls1, t2_dls2,
                     t2_pr, t2_np, t2_h, t2_w, t2_nf, t2_fps, t2_st, t2_sd,
                     t2_qu, t2_of, t2_en, t2_im, t2_if, t2_is],
                    [t2_vid, t2_log])

            # ══════════════════════════════════════════════════════════════
            # 3. One Stage (Prototype)
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🔬 One Stage"):
                gr.HTML('<div class="pipe-info"><strong>TI2VidOneStagePipeline</strong>'
                        ' – Single-stage generation, no upsampling. Primarily for quick'
                        ' prototyping and educational purposes. Output at ~512×768.'
                        ' No distilled LoRA or upsampler needed.</div>')
                with gr.Accordion("⚙️  Model Paths", open=True):
                    with gr.Row():
                        t3_ck = gr.Textbox(label="Checkpoint (.safetensors)", value=ENV["checkpoint"],
                                           placeholder="/models/ltx-2.3-22b-dev.safetensors")
                        t3_gm = gr.Textbox(label="Gemma Root Directory", value=ENV["gemma"],
                                           placeholder="/models/gemma")
                t3_pr, t3_np = _prompt_row()
                t3_h, t3_w, t3_nf, t3_fps = _vid_params(h=512, w=768)
                t3_st, t3_sd, t3_qu, t3_of, t3_en = _adv(30)
                t3_im, t3_if, t3_is = _img_cond()
                t3_btn = _gen_btn("🔬  Generate (One Stage)")
                t3_vid, t3_log = _output()
                t3_btn.click(run_one_stage,
                    [t3_ck, t3_gm, t3_pr, t3_np,
                     t3_h, t3_w, t3_nf, t3_fps, t3_st, t3_sd,
                     t3_qu, t3_of, t3_en, t3_im, t3_if, t3_is],
                    [t3_vid, t3_log])

            # ══════════════════════════════════════════════════════════════
            # 4. Distilled (Fast)
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("⚡ Distilled"):
                gr.HTML('<div class="pipe-info"><strong>DistilledPipeline</strong>'
                        ' – Fastest inference using a fully distilled model with 8 predefined'
                        ' sigma steps (stage 1) + 3 steps (stage 2). No guidance scales needed.'
                        ' Uses distilled checkpoint directly.</div>')
                with gr.Accordion("⚙️  Model Paths", open=True):
                    with gr.Row():
                        t4_dc = gr.Textbox(label="Distilled Checkpoint (.safetensors)",
                                           value=ENV["distilled_ckpt"],
                                           placeholder="/models/ltx-2.3-22b-distilled-1.1.safetensors")
                        t4_gm = gr.Textbox(label="Gemma Root Directory", value=ENV["gemma"],
                                           placeholder="/models/gemma")
                    t4_up = gr.Textbox(label="Spatial Upsampler (.safetensors)",
                                       value=ENV["upsampler"],
                                       placeholder="/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
                t4_pr = gr.Textbox(label="Prompt", lines=4,
                                   placeholder="A futuristic city at night, neon reflections...")
                t4_h, t4_w, t4_nf, t4_fps = _vid_params()
                t4_sd, t4_qu, t4_of, t4_en = _adv_no_steps()
                t4_im, t4_if, t4_is = _img_cond()
                t4_btn = _gen_btn("⚡  Generate (Distilled Fast)")
                t4_vid, t4_log = _output()
                t4_btn.click(run_distilled,
                    [t4_dc, t4_gm, t4_up, t4_pr,
                     t4_h, t4_w, t4_nf, t4_fps, t4_sd,
                     t4_qu, t4_of, t4_en, t4_im, t4_if, t4_is],
                    [t4_vid, t4_log])

            # ══════════════════════════════════════════════════════════════
            # 5. IC-LoRA (Video-to-Video)
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🎭 IC-LoRA (V2V)"):
                gr.HTML('<div class="pipe-info"><strong>ICLoraPipeline</strong>'
                        ' – Video-to-video and image-to-video transformations with In-Context LoRA.'
                        ' Conditions on reference videos (depth maps, poses, edges, etc.).'
                        ' Uses distilled checkpoint only.</div>')
                t5_dc, t5_gm = _paths_distilled_only()
                with gr.Accordion("⚙️  More Model Paths", open=True):
                    with gr.Row():
                        t5_up = gr.Textbox(label="Spatial Upsampler (.safetensors)",
                                           value=ENV["upsampler"])
                        t5_ic = gr.Textbox(label="IC-LoRA (.safetensors)",
                                           placeholder="/models/ltx-2.3-22b-ic-lora-union-control.safetensors")
                        t5_ics = gr.Slider(label="IC-LoRA Strength", minimum=0.0, maximum=1.5,
                                           step=0.05, value=1.0)
                with gr.Accordion("🎬  Video Conditioning (required)", open=True):
                    gr.Markdown("_The reference video(s) that guide the transformation (e.g. depth map, pose video)._")
                    with gr.Row():
                        t5_vc1 = gr.Textbox(label="Reference Video 1 (required)",
                                            placeholder="/path/to/control_video.mp4")
                        t5_vs1 = gr.Slider(label="Conditioning Strength 1",
                                           minimum=0.1, maximum=1.5, step=0.05, value=1.0)
                    with gr.Row():
                        t5_vc2 = gr.Textbox(label="Reference Video 2 (optional)",
                                            placeholder="Leave empty if not needed")
                        t5_vs2 = gr.Slider(label="Conditioning Strength 2",
                                           minimum=0.1, maximum=1.5, step=0.05, value=1.0)
                t5_pr = gr.Textbox(label="Prompt", lines=4,
                                   placeholder="Describe the target video style/content...")
                t5_h, t5_w, t5_nf, t5_fps = _vid_params()
                t5_sd, t5_qu, t5_of, t5_en = _adv_no_steps()
                t5_im, t5_if, t5_is = _img_cond()
                t5_skip = gr.Checkbox(label="Skip Stage 2 (half resolution, faster)", value=False)
                t5_btn = _gen_btn("🎭  Generate (IC-LoRA V2V)")
                t5_vid, t5_log = _output()
                t5_btn.click(run_ic_lora,
                    [t5_dc, t5_gm, t5_up, t5_ic, t5_ics,
                     t5_pr, t5_h, t5_w, t5_nf, t5_fps, t5_sd,
                     t5_qu, t5_of, t5_en,
                     t5_vc1, t5_vs1, t5_vc2, t5_vs2,
                     t5_im, t5_if, t5_is, t5_skip],
                    [t5_vid, t5_log])

            # ══════════════════════════════════════════════════════════════
            # 6. Keyframe Interpolation
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🖼️ Keyframe Interp"):
                gr.HTML('<div class="pipe-info"><strong>KeyframeInterpolationPipeline</strong>'
                        ' – Generate smooth video by interpolating between keyframe images.'
                        ' All uploaded images become guiding latents (additive conditioning)'
                        ' for smooth transitions. Supports up to 4 keyframes.</div>')
                t6_ck, t6_gm, t6_up, t6_dl, t6_dls = _paths_two_stage()
                t6_pr, t6_np = _prompt_row()
                t6_h, t6_w, t6_nf, t6_fps = _vid_params()
                t6_st, t6_sd, t6_qu, t6_of, t6_en = _adv(30)
                with gr.Accordion("🖼️  Keyframe Images (at least 1 required)", open=True):
                    gr.Markdown("_Specify image path, which video frame index it targets, and its strength._")
                    with gr.Row():
                        t6_k1p = gr.Textbox(label="Keyframe 1 Path", placeholder="/path/to/first_frame.jpg")
                        t6_k1f = gr.Number(label="Frame Index", value=0, precision=0)
                        t6_k1s = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.05, value=1.0)
                    with gr.Row():
                        t6_k2p = gr.Textbox(label="Keyframe 2 Path", placeholder="/path/to/middle.jpg")
                        t6_k2f = gr.Number(label="Frame Index", value=60, precision=0)
                        t6_k2s = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.05, value=1.0)
                    with gr.Row():
                        t6_k3p = gr.Textbox(label="Keyframe 3 Path (optional)")
                        t6_k3f = gr.Number(label="Frame Index", value=90, precision=0)
                        t6_k3s = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.05, value=1.0)
                    with gr.Row():
                        t6_k4p = gr.Textbox(label="Keyframe 4 Path (optional)")
                        t6_k4f = gr.Number(label="Frame Index", value=120, precision=0)
                        t6_k4s = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.05, value=1.0)
                t6_btn = _gen_btn("🖼️  Generate (Keyframe Interpolation)")
                t6_vid, t6_log = _output()
                t6_btn.click(run_keyframe,
                    [t6_ck, t6_gm, t6_up, t6_dl, t6_dls,
                     t6_pr, t6_np, t6_h, t6_w, t6_nf, t6_fps, t6_st, t6_sd,
                     t6_qu, t6_of, t6_en,
                     t6_k1p, t6_k1f, t6_k1s,
                     t6_k2p, t6_k2f, t6_k2s,
                     t6_k3p, t6_k3f, t6_k3s,
                     t6_k4p, t6_k4f, t6_k4s],
                    [t6_vid, t6_log])

            # ══════════════════════════════════════════════════════════════
            # 7. Audio-to-Video
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🎵 Audio-to-Video"):
                gr.HTML('<div class="pipe-info"><strong>A2VidPipelineTwoStage</strong>'
                        ' – Generate video driven by an input audio file. Stage 1 denoises'
                        ' video with audio frozen; Stage 2 upsamples 2×. Original audio'
                        ' is preserved in the output without re-decoding.</div>')
                t7_ck, t7_gm, t7_up, t7_dl, t7_dls = _paths_two_stage()
                with gr.Accordion("🎵  Audio Input (required)", open=True):
                    with gr.Row():
                        t7_ap = gr.Textbox(label="Audio File Path (required)",
                                           placeholder="/path/to/audio.wav  (.mp3 / .flac also supported)")
                        t7_as = gr.Number(label="Audio Start Time (s)", value=0.0,
                                          info="0 = beginning of audio")
                        t7_am = gr.Number(label="Max Duration (s)", value=0.0,
                                          info="0 = use full audio (trimmed to match video length)")
                t7_pr, t7_np = _prompt_row()
                t7_h, t7_w, t7_nf, t7_fps = _vid_params()
                t7_st, t7_sd, t7_qu, t7_of, t7_en = _adv(30)
                t7_im, t7_if, t7_is = _img_cond()
                t7_btn = _gen_btn("🎵  Generate (Audio-to-Video)")
                t7_vid, t7_log = _output()
                t7_btn.click(run_a2vid,
                    [t7_ck, t7_gm, t7_up, t7_dl, t7_dls,
                     t7_pr, t7_np, t7_h, t7_w, t7_nf, t7_fps, t7_st, t7_sd,
                     t7_qu, t7_of, t7_en, t7_ap, t7_as, t7_am,
                     t7_im, t7_if, t7_is],
                    [t7_vid, t7_log])

            # ══════════════════════════════════════════════════════════════
            # 8. Retake (Video Editing)
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("✂️ Retake"):
                gr.HTML('<div class="pipe-info"><strong>RetakePipeline</strong>'
                        ' – Regenerate a specific time window of an existing video while'
                        ' keeping everything outside that window unchanged. Source video'
                        ' frame count must be 8k+1 and resolution divisible by 32.</div>')
                t8_dc, t8_gm = _paths_distilled_only()
                with gr.Accordion("📹  Source Video & Time Window", open=True):
                    t8_vp = gr.Textbox(label="Source Video Path (required)",
                                       placeholder="/path/to/source.mp4",
                                       info="Frame count must be 8k+1; resolution ÷ 32")
                    with gr.Row():
                        t8_ts = gr.Number(label="Start Time (s)", value=1.0)
                        t8_te = gr.Number(label="End Time (s)", value=3.0)
                t8_pr, t8_np = _prompt_row(lines=3)
                with gr.Accordion("🔧  Advanced", open=False):
                    with gr.Row():
                        t8_st = gr.Slider(label="Inference Steps", minimum=4, maximum=60,
                                          step=1, value=40)
                        t8_sd = gr.Number(label="Seed", value=42, precision=0)
                    with gr.Row():
                        t8_qu = gr.Dropdown(label="Quantization",
                                            choices=["none", "fp8-cast", "fp8-scaled-mm"], value="fp8-cast")
                        t8_of = gr.Dropdown(label="Offload",
                                            choices=["none", "cpu", "disk"], value="none")
                t8_btn = _gen_btn("✂️  Run Retake")
                t8_vid, t8_log = _output()
                t8_btn.click(run_retake,
                    [t8_dc, t8_gm, t8_pr, t8_np, t8_st, t8_sd,
                     t8_qu, t8_of, t8_vp, t8_ts, t8_te],
                    [t8_vid, t8_log])

            # ══════════════════════════════════════════════════════════════
            # 9. HDR IC-LoRA
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🌟 HDR IC-LoRA"):
                gr.HTML('<div class="pipe-info"><strong>HDRICLoraPipeline</strong>'
                        ' – Video-to-video with HDR output (linear float / EXR frames'
                        ' via ARRI LogC3 inverse decode). No Gemma needed — uses'
                        ' pre-computed text embeddings. Also exports H.264 preview.'
                        ' LoRA: <em>LTX-2.3-22b-IC-LoRA-HDR</em> on HuggingFace.</div>')
                with gr.Accordion("⚙️  Model Paths", open=True):
                    with gr.Row():
                        t9_dc = gr.Textbox(label="Distilled Checkpoint (.safetensors)",
                                           value=ENV["distilled_ckpt"])
                        t9_up = gr.Textbox(label="Spatial Upsampler (.safetensors)",
                                           value=ENV["upsampler"])
                    with gr.Row():
                        t9_hl = gr.Textbox(label="HDR IC-LoRA (.safetensors)",
                                           placeholder="/models/ltx-2.3-22b-ic-lora-hdr.safetensors")
                        t9_te = gr.Textbox(label="Pre-computed Text Embeddings (.safetensors)",
                                           placeholder="/models/hdr_scene_embeddings.safetensors",
                                           info="Computed externally via PromptEncoder; no Gemma needed here")
                with gr.Accordion("📹  Input & Output", open=True):
                    with gr.Row():
                        t9_inp = gr.Textbox(label="Input Video or Directory",
                                            placeholder="/path/to/input.mp4  (or /path/to/video_dir/)")
                        t9_nf = gr.Slider(label="Frames (8k+1)", minimum=9, maximum=257,
                                          step=8, value=161)
                with gr.Accordion("🔧  Advanced Parameters", open=False):
                    with gr.Row():
                        t9_sd = gr.Number(label="Seed", value=10, precision=0)
                        t9_of = gr.Dropdown(label="Offload", choices=["none", "cpu", "disk"], value="none",
                                            info="Note: offload disables fp8 quantization")
                        t9_st = gr.Slider(label="Spatial Tile (px)", minimum=512, maximum=2048,
                                          step=128, value=1536,
                                          info="Reduce to 768 on lower-VRAM GPUs")
                    with gr.Row():
                        t9_sm = gr.Checkbox(label="Skip MP4 preview (EXR only)", value=False)
                        t9_eh = gr.Checkbox(label="EXR float16 (smaller files)", value=False)
                        t9_hq = gr.Checkbox(label="High-quality HDR (2x slower)", value=False)
                t9_btn = _gen_btn("🌟  Generate (HDR IC-LoRA)")
                t9_vid, t9_log = _output()
                t9_btn.click(run_hdr_ic_lora,
                    [t9_dc, t9_up, t9_hl, t9_te,
                     t9_inp, t9_nf, t9_sd, t9_of,
                     t9_sm, t9_eh, t9_hq, t9_st],
                    [t9_vid, t9_log])

            # ══════════════════════════════════════════════════════════════
            # 10. LipDub
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("👄 LipDub"):
                gr.HTML('<div class="pipe-info"><strong>LipDubPipeline</strong>'
                        ' – Lip dubbing and rephrasing while preserving speaker identity.'
                        ' Uses IC-LoRA + audio reference conditioning on a distilled model.'
                        ' Frame count and FPS are derived from the reference video automatically.'
                        ' LoRA: <em>LTX-2.3-22b-IC-LoRA-LipDub</em> on HuggingFace.</div>')
                with gr.Accordion("⚙️  Model Paths", open=True):
                    with gr.Row():
                        t10_dc = gr.Textbox(label="Distilled Checkpoint (.safetensors)",
                                            value=ENV["distilled_ckpt"])
                        t10_gm = gr.Textbox(label="Gemma Root Directory", value=ENV["gemma"])
                    with gr.Row():
                        t10_up = gr.Textbox(label="Spatial Upsampler (.safetensors)",
                                            value=ENV["upsampler"])
                        t10_il = gr.Textbox(label="LipDub IC-LoRA (.safetensors)",
                                            placeholder="/models/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors")
                        t10_ils = gr.Slider(label="IC-LoRA Strength", minimum=0.0, maximum=1.5,
                                            step=0.05, value=1.0)
                with gr.Accordion("📹  Reference Video", open=True):
                    t10_rv = gr.Textbox(label="Reference Video Path (required)",
                                        placeholder="/path/to/source_speaker.mp4",
                                        info="Provides face identity + audio track. Frame count snapped to 8k+1 automatically.")
                    t10_rs = gr.Slider(label="Reference Strength", minimum=0.0, maximum=1.5,
                                       step=0.05, value=1.0,
                                       info="Controls how strongly the reference face/audio is used")
                t10_pr = gr.Textbox(label="Prompt (describe the new speech / lip movement)",
                                    lines=3, placeholder="A person saying 'Hello world' clearly and naturally...")
                with gr.Row():
                    t10_h, t10_w = (gr.Slider(label="Height (px)", minimum=256, maximum=1920,
                                               step=64, value=1024),
                                    gr.Slider(label="Width (px)",  minimum=256, maximum=1920,
                                               step=64, value=1536))
                t10_sd, t10_qu, t10_of, _ = _adv_no_steps()
                t10_btn = _gen_btn("👄  Generate (LipDub)")
                t10_vid, t10_log = _output()
                t10_btn.click(run_lipdub,
                    [t10_dc, t10_gm, t10_up, t10_il, t10_ils,
                     t10_pr, t10_h, t10_w, t10_sd, t10_qu, t10_of,
                     t10_rv, t10_rs],
                    [t10_vid, t10_log])

            # ══════════════════════════════════════════════════════════════
            # 11. Setup Guide
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("📖 Setup Guide"):
                gr.Markdown("""
## SimplePod Setup Guide

### 1 · Install dependencies

```bash
cd /workspace/LTX-2
uv sync --frozen && source .venv/bin/activate
# or: pip install -e packages/ltx-core packages/ltx-pipelines && pip install gradio
```

---

### 2 · Download models from HuggingFace

```bash
# Main model (pick one)
wget https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors

# Distilled (fastest – needed for Distilled / IC-LoRA / Retake / LipDub tabs)
wget https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors

# Spatial upsampler (all two-stage pipelines)
wget https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

# Distilled LoRA (TI2Vid / A2Vid / Keyframe tabs)
wget https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors

# Gemma text encoder
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir ./models/gemma
```

#### Optional LoRAs

```bash
# IC-LoRA (union control: depth/pose/edges) → IC-LoRA tab
huggingface-cli download Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control \
    --local-dir ./models/ic_lora_union

# LipDub IC-LoRA → LipDub tab
wget https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub/resolve/main/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors

# HDR IC-LoRA (+ pre-computed embeddings) → HDR tab
huggingface-cli download Lightricks/LTX-2.3-22b-IC-LoRA-HDR --local-dir ./models/hdr_lora
```

---

### 3 · Set environment variables (optional)

Pre-fill all path inputs in the UI:

```bash
export LTX_CHECKPOINT_PATH="/workspace/models/ltx-2.3-22b-dev.safetensors"
export LTX_DISTILLED_CHECKPOINT="/workspace/models/ltx-2.3-22b-distilled-1.1.safetensors"
export LTX_GEMMA_ROOT="/workspace/models/gemma"
export LTX_UPSAMPLER_PATH="/workspace/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
export LTX_DISTILLED_LORA_PATH="/workspace/models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
export LTX_OUTPUT_DIR="/workspace/outputs"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

---

### 4 · Launch

```bash
bash launch.sh              # port 7860, localhost
bash launch.sh --share      # public Gradio URL
bash launch.sh --port 8080  # custom port
```

---

### Pipeline Quick Reference

| Tab | Pipeline | Needs | Speed | Quality |
|-----|----------|-------|-------|---------|
| ⭐ TI2Vid | TI2VidTwoStagesPipeline | ckpt + lora + upsamp | ★★★ | ★★★★★ |
| 🏆 TI2Vid HQ | TI2VidTwoStagesHQPipeline | ckpt + lora + upsamp | ★★★ | ★★★★★ |
| 🔬 One Stage | TI2VidOneStagePipeline | ckpt only | ★★★★ | ★★★ |
| ⚡ Distilled | DistilledPipeline | dist-ckpt + upsamp | ★★★★★ | ★★★★ |
| 🎭 IC-LoRA | ICLoraPipeline | dist-ckpt + upsamp + ic-lora | ★★★★ | ★★★★ |
| 🖼️ Keyframe | KeyframeInterpolationPipeline | ckpt + lora + upsamp | ★★★ | ★★★★★ |
| 🎵 A2Vid | A2VidPipelineTwoStage | ckpt + lora + upsamp | ★★★ | ★★★★★ |
| ✂️ Retake | RetakePipeline | dist-ckpt | ★★★★ | ★★★★ |
| 🌟 HDR | HDRICLoraPipeline | dist-ckpt + upsamp + hdr-lora | ★★★ | ★★★★★ |
| 👄 LipDub | LipDubPipeline | dist-ckpt + upsamp + lipdub-lora | ★★★★ | ★★★★ |

### GPU Recommendations

| GPU | VRAM | Recommended setting |
|-----|------|---------------------|
| H100 / A100 80GB | 80 GB | No quantization, no offload |
| A100 40GB | 40 GB | `fp8-cast` |
| A10G | 24 GB | `fp8-cast` + `offload cpu` |
| L4 / RTX 4090 | 24 GB | `fp8-cast` + `offload cpu` |

### Prompting Tips

> **Structure:** action → movements → appearance → background → camera → lighting → color

```
A golden retriever runs across a sunlit meadow toward the camera, tongue out
and ears flapping. Slow motion. Tall grass sways in the breeze. Shallow depth
of field, soft bokeh background. Warm golden hour lighting. Ground-level
tracking shot. Vivid warm colors.
```
                """)

        gr.HTML("""
        <div style="text-align:center;padding:20px 0 8px;color:#94a3b8;font-size:.82rem;">
          LTX-2 by <a href="https://ltx.io" style="color:#6366f1">Lightricks</a> ·
          <a href="https://huggingface.co/Lightricks/LTX-2.3" style="color:#6366f1">HuggingFace</a> ·
          <a href="https://github.com/Lightricks/LTX-2" style="color:#6366f1">GitHub</a>
        </div>
        """)

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    """Parse CLI arguments and launch the Gradio server."""
    parser = argparse.ArgumentParser(description="LTX-2 Web UI")
    parser.add_argument("--host",  default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",  type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--no-queue", action="store_true",
                        help="Disable Gradio queue (breaks streaming)")
    args = parser.parse_args()

    demo = build_ui()
    if not args.no_queue:
        demo.queue(max_size=5)
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
