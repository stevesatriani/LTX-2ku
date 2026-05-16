"""
LTX-2 Model Downloader
Gradio UI to download all required and optional LTX-2 models from HuggingFace.
Uses huggingface_hub for reliable, resumable downloads.

Download structure inside <base_dir>:
    models/
    ├── checkpoints/      ← LTX-2.3 dev & distilled checkpoints (~41 GB each)
    ├── components/       ← upsampler x2/x1.5, temporal upsampler, distilled LoRA
    ├── text_encoder/     ← Gemma 3 12B quantized (required by all pipelines)
    └── loras/            ← IC-LoRAs, camera control LoRAs, HDR, LipDub

NOTE: The directory packages/ltx-core/src/ltx_core/model/ contains Python source
      code (audio_vae, transformer, video_vae…) — do NOT put weights there.
      Use the models/ directory at the project root instead.

Usage:
    python download_models.py
    python download_models.py --port 7861 --share
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

# ─── Catalog ──────────────────────────────────────────────────────────────────
# subdir  : folder inside <base_dir> where this model lands
# file    : filename on HF; None = full repo download
# is_repo : True when downloading an entire HF repo (no single file)

CATALOG = {

    "checkpoints": {
        "label": "🧠  Main Checkpoints",
        "note": (
            "Large files (~41 GB each). Choose at least one. "
            "Dev = best quality (TI2Vid / HQ / A2Vid), "
            "Distilled = fastest (Distilled / IC-LoRA / Retake / LipDub / HDR)."
        ),
        "models": [
            {
                "id":        "ltx_dev",
                "name":      "LTX-2.3 Dev (22B) – full model",
                "repo":      "Lightricks/LTX-2.3",
                "file":      "ltx-2.3-22b-dev.safetensors",
                "subdir":    "checkpoints",
                "is_repo":   False,
                "size":      "~41 GB",
                "desc":      "Best quality. Required for TI2Vid, TI2Vid HQ, One Stage, Keyframe, A2Vid tabs.",
                "pipelines": ["TI2Vid", "TI2Vid HQ", "One Stage", "Keyframe", "A2Vid"],
                "default":   True,
            },
            {
                "id":        "ltx_distilled",
                "name":      "LTX-2.3 Distilled (22B) – fast model",
                "repo":      "Lightricks/LTX-2.3",
                "file":      "ltx-2.3-22b-distilled-1.1.safetensors",
                "subdir":    "checkpoints",
                "is_repo":   False,
                "size":      "~41 GB",
                "desc":      "Fastest inference. Required for Distilled, IC-LoRA, Retake, LipDub, HDR tabs.",
                "pipelines": ["Distilled", "IC-LoRA", "Retake", "LipDub", "HDR"],
                "default":   True,
            },
        ],
    },

    "components": {
        "label": "⚙️  Required Components",
        "note": (
            "Essential support models. "
            "Upsampler x2 + Distilled LoRA + Gemma are needed by most pipelines."
        ),
        "models": [
            {
                "id":        "upsampler_x2",
                "name":      "Spatial Upsampler ×2  (recommended)",
                "repo":      "Lightricks/LTX-2.3",
                "file":      "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                "subdir":    "components",
                "is_repo":   False,
                "size":      "~600 MB",
                "desc":      "2× upsampling for Stage 2. Required by all two-stage pipelines.",
                "pipelines": ["TI2Vid", "TI2Vid HQ", "Distilled", "IC-LoRA", "Keyframe", "A2Vid", "LipDub", "HDR"],
                "default":   True,
            },
            {
                "id":        "upsampler_x1_5",
                "name":      "Spatial Upsampler ×1.5",
                "repo":      "Lightricks/LTX-2.3",
                "file":      "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
                "subdir":    "components",
                "is_repo":   False,
                "size":      "~600 MB",
                "desc":      "Alternative to ×2 upsampler for lower VRAM or different aspect ratios.",
                "pipelines": ["Two-stage (alternative)"],
                "default":   False,
            },
            {
                "id":        "temporal_upsampler",
                "name":      "Temporal Upsampler ×2",
                "repo":      "Lightricks/LTX-2.3",
                "file":      "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
                "subdir":    "components",
                "is_repo":   False,
                "size":      "~600 MB",
                "desc":      "Frame-count upsampling. Supported by model; reserved for future pipeline implementations.",
                "pipelines": ["Future pipelines"],
                "default":   False,
            },
            {
                "id":        "distilled_lora",
                "name":      "Distilled LoRA 384",
                "repo":      "Lightricks/LTX-2.3",
                "file":      "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                "subdir":    "components",
                "is_repo":   False,
                "size":      "~1.5 GB",
                "desc":      "Required for Stage 2 refinement in TI2Vid, TI2Vid HQ, Keyframe, A2Vid tabs.",
                "pipelines": ["TI2Vid", "TI2Vid HQ", "Keyframe", "A2Vid"],
                "default":   True,
            },
            {
                "id":        "gemma",
                "name":      "Gemma 3  Text Encoder (12B quantized – full repo)",
                "repo":      "google/gemma-3-12b-it-qat-q4_0-unquantized",
                "file":      None,
                "subdir":    "text_encoder",
                "is_repo":   True,
                "size":      "~7 GB",
                "desc":      (
                    "Text encoder required by all pipelines except HDR. "
                    "Gated model – requires HF token + license acceptance at huggingface.co."
                ),
                "pipelines": ["All pipelines except HDR"],
                "default":   True,
            },
        ],
    },

    "ic_loras": {
        "label": "🎭  IC-LoRAs",
        "note": (
            "Specialized adapters for the IC-LoRA, HDR, and LipDub tabs. "
            "Download only what you plan to use."
        ),
        "models": [
            {
                "id":        "ic_lora_union",
                "name":      "IC-LoRA Union Control  (depth / pose / edges)",
                "repo":      "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                "file":      "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                "subdir":    "loras",
                "is_repo":   False,
                "size":      "~400 MB",
                "desc":      "Versatile control signal: depth maps, human poses, edge maps.",
                "pipelines": ["IC-LoRA"],
                "default":   False,
            },
            {
                "id":        "ic_lora_motion",
                "name":      "IC-LoRA Motion Track Control",
                "repo":      "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
                "file":      "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
                "subdir":    "loras",
                "is_repo":   False,
                "size":      "~400 MB",
                "desc":      "Control generation using motion tracking data.",
                "pipelines": ["IC-LoRA"],
                "default":   False,
            },
            {
                "id":        "ic_lora_detailer",
                "name":      "IC-LoRA Detailer  (19B model only)",
                "repo":      "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
                "file":      "ltx-2-19b-ic-lora-detailer.safetensors",
                "subdir":    "loras",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Fine-detail enhancement. Requires LTX-2 19B base model (not 22B).",
                "pipelines": ["IC-LoRA (19B only)"],
                "default":   False,
            },
            {
                "id":        "ic_lora_pose",
                "name":      "IC-LoRA Pose Control  (19B model only)",
                "repo":      "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control",
                "file":      "ltx-2-19b-ic-lora-pose-control.safetensors",
                "subdir":    "loras",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Full-body pose conditioning. Requires LTX-2 19B base model.",
                "pipelines": ["IC-LoRA (19B only)"],
                "default":   False,
            },
            {
                "id":        "ic_lora_hdr",
                "name":      "IC-LoRA HDR  (full repo – includes text embeddings)",
                "repo":      "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
                "file":      None,
                "subdir":    "loras",
                "is_repo":   True,
                "size":      "~500 MB",
                "desc":      "HDR video output via LogC3 inverse decode. Repo includes pre-computed text embeddings needed by the HDR tab.",
                "pipelines": ["HDR IC-LoRA"],
                "default":   False,
            },
            {
                "id":        "ic_lora_lipdub",
                "name":      "IC-LoRA LipDub",
                "repo":      "Lightricks/LTX-2.3-22b-IC-LoRA-LipDub",
                "file":      "ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors",
                "subdir":    "loras",
                "is_repo":   False,
                "size":      "~400 MB",
                "desc":      "Lip dubbing with speaker identity preservation. Use with LipDub tab.",
                "pipelines": ["LipDub"],
                "default":   False,
            },
        ],
    },

    "camera_loras": {
        "label": "🎥  Camera Control LoRAs  (19B model only)",
        "note": "Cinematic camera movements. Each LoRA controls one motion type. Requires LTX-2 19B base model.",
        "models": [
            {
                "id":        "cam_dolly_in",
                "name":      "Camera – Dolly In",
                "repo":      "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
                "file":      "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
                "subdir":    "loras/camera",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Push camera toward the subject (dolly in).",
                "pipelines": ["IC-LoRA (19B)"],
                "default":   False,
            },
            {
                "id":        "cam_dolly_out",
                "name":      "Camera – Dolly Out",
                "repo":      "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
                "file":      "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
                "subdir":    "loras/camera",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Pull camera away from the subject (dolly out).",
                "pipelines": ["IC-LoRA (19B)"],
                "default":   False,
            },
            {
                "id":        "cam_dolly_left",
                "name":      "Camera – Dolly Left",
                "repo":      "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
                "file":      "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
                "subdir":    "loras/camera",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Lateral dolly movement to the left.",
                "pipelines": ["IC-LoRA (19B)"],
                "default":   False,
            },
            {
                "id":        "cam_dolly_right",
                "name":      "Camera – Dolly Right",
                "repo":      "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
                "file":      "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
                "subdir":    "loras/camera",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Lateral dolly movement to the right.",
                "pipelines": ["IC-LoRA (19B)"],
                "default":   False,
            },
            {
                "id":        "cam_jib_up",
                "name":      "Camera – Jib Up",
                "repo":      "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
                "file":      "ltx-2-19b-lora-camera-control-jib-up.safetensors",
                "subdir":    "loras/camera",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Vertical crane/jib movement upward.",
                "pipelines": ["IC-LoRA (19B)"],
                "default":   False,
            },
            {
                "id":        "cam_jib_down",
                "name":      "Camera – Jib Down",
                "repo":      "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
                "file":      "ltx-2-19b-lora-camera-control-jib-down.safetensors",
                "subdir":    "loras/camera",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Vertical crane/jib movement downward.",
                "pipelines": ["IC-LoRA (19B)"],
                "default":   False,
            },
            {
                "id":        "cam_static",
                "name":      "Camera – Static  (locked off)",
                "repo":      "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
                "file":      "ltx-2-19b-lora-camera-control-static.safetensors",
                "subdir":    "loras/camera",
                "is_repo":   False,
                "size":      "~300 MB",
                "desc":      "Enforce a completely static, locked-off camera.",
                "pipelines": ["IC-LoRA (19B)"],
                "default":   False,
            },
        ],
    },
}

# ─── Flat lookup id → entry ────────────────────────────────────────────────────
_ALL: dict[str, dict] = {}
for _grp in CATALOG.values():
    for _m in _grp["models"]:
        _ALL[_m["id"]] = _m


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _dest(m: dict, base: str) -> Path:
    """Return the directory where a model's file will be saved."""
    return Path(base) / m["subdir"]


def _final_path(m: dict, base: str) -> Path:
    """Return the expected full path of a downloaded model (file or directory)."""
    d = _dest(m, base)
    if m["is_repo"]:
        repo_name = m["repo"].split("/")[-1]
        return d / repo_name
    return d / m["file"]


def _exists(m: dict, base: str) -> bool:
    """Check if a model is already present on disk."""
    p = _final_path(m, base)
    if m["is_repo"]:
        return p.is_dir() and any(p.glob("*.safetensors"))
    return p.is_file()


def _path_ref(m: dict, base: str) -> str:
    """Return the human-readable path for the path reference accordion."""
    return str(_final_path(m, base))


# ─── Download logic ────────────────────────────────────────────────────────────

def download_models(base_dir: str, hf_token: str, *checkbox_lists):
    """
    Download selected models to organised subdirectories inside base_dir.
    Streams live huggingface_hub output to the Gradio log textbox.
    Yields (log_text,) for real-time updates.
    """
    selected: list[str] = [mid for lst in checkbox_lists if lst for mid in lst]

    if not selected:
        yield "⚠️  No models selected. Tick at least one checkbox."
        return
    if not base_dir or not base_dir.strip():
        yield "⚠️  Please set the download directory."
        return

    base = base_dir.strip()
    env = {**os.environ}
    if hf_token and hf_token.strip():
        env["HF_TOKEN"] = hf_token.strip()

    lines: list[str] = []
    total = len(selected)

    def emit(msg: str = ""):
        lines.append(msg + "\n")

    for idx, mid in enumerate(selected, 1):
        m = _ALL.get(mid)
        if not m:
            continue

        dest_dir = _dest(m, base)
        dest_dir.mkdir(parents=True, exist_ok=True)

        emit(f"\n{'─'*62}")
        emit(f"[{idx}/{total}]  {m['name']}  ({m['size']})")
        emit(f"        repo : {m['repo']}")
        if m["file"]:
            emit(f"        file : {m['file']}")
        emit(f"        dest : {dest_dir}")

        if _exists(m, base):
            emit("        ✅  Already downloaded – skipping.\n")
            yield "".join(lines)
            continue

        # Build download command using the Python API inline script.
        # This avoids depending on huggingface_hub.commands (not present in older
        # versions) while still streaming real-time tqdm output via subprocess.
        token_repr = repr(hf_token.strip() if hf_token and hf_token.strip() else None)

        if not m["is_repo"]:
            script = (
                "import sys; from huggingface_hub import hf_hub_download; "
                f"p = hf_hub_download("
                f"repo_id={repr(m['repo'])}, filename={repr(m['file'])}, "
                f"local_dir={repr(str(dest_dir))}, token={token_repr}, resume_download=True); "
                "print('Saved to:', p)"
            )
        else:
            repo_name = m["repo"].split("/")[-1]
            script = (
                "import sys; from huggingface_hub import snapshot_download; "
                f"p = snapshot_download("
                f"repo_id={repr(m['repo'])}, "
                f"local_dir={repr(str(dest_dir / repo_name))}, token={token_repr}); "
                "print('Saved to:', p)"
            )

        cmd = [sys.executable, "-c", script]

        emit(f"\n        $ python -c \"huggingface_hub ... {m['repo']}\"\n")
        yield "".join(lines)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in iter(proc.stdout.readline, ""):
            lines.append(line)
            yield "".join(lines)
        proc.wait()

        if proc.returncode == 0:
            emit(f"\n        ✅  Saved to: {_final_path(m, base)}\n")
        else:
            emit(f"\n        ❌  Failed (exit code {proc.returncode}).\n"
                 "            Check HF token / network / disk space.\n")
        yield "".join(lines)

    # ── Final summary ──────────────────────────────────────────────────────
    emit(f"\n{'='*62}")
    emit("Download session complete!")
    emit(f"Base directory : {base}\n")
    for mid in selected:
        m = _ALL.get(mid)
        if m:
            icon = "✅" if _exists(m, base) else "❌"
            emit(f"  {icon}  {m['name']}")
            emit(f"       → {_final_path(m, base)}")
    yield "".join(lines)


def check_status(base_dir: str, *checkbox_lists) -> str:
    """Show which selected models are already present on disk."""
    selected = [mid for lst in checkbox_lists if lst for mid in lst]
    if not selected:
        return "No models selected."
    base = base_dir.strip() if base_dir else ""
    rows = []
    for mid in selected:
        m = _ALL.get(mid)
        if m:
            icon = "✅" if _exists(m, base) else "⬜"
            rows.append(f"  {icon}  {m['name']}  ({m['size']})")
            rows.append(f"       {_final_path(m, base)}")
    return "Status check:\n\n" + "\n".join(rows)


def _env_block(base_dir: str) -> str:
    """Generate shell export lines for all default=True models."""
    b = base_dir or "./models"
    lines = [
        f'MODEL_DIR="{b}"',
        "",
        f'export LTX_CHECKPOINT_PATH="$MODEL_DIR/checkpoints/ltx-2.3-22b-dev.safetensors"',
        f'export LTX_DISTILLED_CHECKPOINT="$MODEL_DIR/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"',
        f'export LTX_GEMMA_ROOT="$MODEL_DIR/text_encoder/gemma-3-12b-it-qat-q4_0-unquantized"',
        f'export LTX_UPSAMPLER_PATH="$MODEL_DIR/components/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"',
        f'export LTX_DISTILLED_LORA_PATH="$MODEL_DIR/components/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"',
        f'export LTX_OUTPUT_DIR="./outputs"',
    ]
    return "\n".join(lines)


# ─── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container {
    max-width: 1100px !important; margin: 0 auto !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
#dl-header {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 55%, #ec4899 100%);
    border-radius: 20px; padding: 26px 36px; margin-bottom: 20px;
    color: white; box-shadow: 0 12px 40px rgba(99,102,241,.35);
}
#dl-header h1 { font-size: 2rem; font-weight: 800; margin: 0 0 4px; }
#dl-header .sub { font-size: .95rem; opacity: .88; margin: 0; }
#dl-header .badges span {
    display: inline-block; background: rgba(255,255,255,.22); border-radius: 20px;
    padding: 3px 12px; font-size: .75rem; font-weight: 600; margin: 8px 4px 0;
}
.tree-box {
    background: #0f172a; color: #94a3b8; border-radius: 12px;
    padding: 16px 20px; font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: .82rem; line-height: 1.8; margin-bottom: 12px;
    border: 1px solid #1e3a5f;
}
.tree-box .hi  { color: #7dd3fc; }
.tree-box .ok  { color: #4ade80; }
.tree-box .dim { color: #475569; }
.warn-box {
    background: #fefce8; border: 1px solid #fde047; border-radius: 10px;
    padding: 10px 14px; font-size: .84rem; color: #92400e; margin-bottom: 12px;
}
.group-hdr {
    font-size: 1rem; font-weight: 700; color: #4f46e5;
    margin: 18px 0 4px; padding-bottom: 6px;
    border-bottom: 2px solid #e0e7ff;
}
.group-note {
    font-size: .83rem; color: #6b7280; margin: 0 0 10px;
    padding: 6px 12px; background: #f8fafc;
    border-left: 3px solid #a5b4fc; border-radius: 0 6px 6px 0;
}
#log-out textarea {
    font-family: 'JetBrains Mono','Courier New',monospace !important;
    font-size: .77rem !important; background: #0f172a !important;
    color: #7dd3fc !important; border: 1px solid #1e3a5f !important;
    border-radius: 12px !important; line-height: 1.6 !important;
}
.dl-btn {
    background: linear-gradient(135deg,#6366f1 0%,#a855f7 100%) !important;
    border: none !important; border-radius: 12px !important;
    font-size: 1.02rem !important; font-weight: 700 !important;
    color: white !important; min-height: 52px !important;
    box-shadow: 0 4px 15px rgba(99,102,241,.35) !important;
    transition: all .25s ease !important;
}
.dl-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(99,102,241,.55) !important; }
.sel-btn {
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: .85rem !important; padding: 7px 14px !important;
}
"""

# ─── UI ───────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:

    def _choices(key: str):
        return [
            (f"{m['name']}  [{m['size']}]  — {', '.join(m['pipelines'])}", m["id"])
            for m in CATALOG[key]["models"]
        ]

    def _defaults(key: str):
        return [m["id"] for m in CATALOG[key]["models"] if m.get("default")]

    def _set_recommended():
        rec = {mid for mid, m in _ALL.items() if m.get("default")}
        return [
            [i for i in [m["id"] for m in CATALOG[k]["models"]] if i in rec]
            for k in ["checkpoints", "components", "ic_loras", "camera_loras"]
        ]

    def _set_all():
        return [[m["id"] for m in CATALOG[k]["models"]]
                for k in ["checkpoints", "components", "ic_loras", "camera_loras"]]

    def _set_none():
        return [[], [], [], []]

    def _update_env(base_dir: str):
        return f"```bash\n{_env_block(base_dir)}\n```"

    with gr.Blocks(
        title="LTX-2 Model Downloader",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            radius_size=gr.themes.sizes.radius_lg,
        ),
        css=CSS,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="dl-header">
          <h1>⬇️  LTX-2 Model Downloader</h1>
          <p class="sub">
            Download all required and optional LTX-2 models from HuggingFace.
            Files are organized into clean subdirectories. Already-downloaded files are skipped.
          </p>
          <div class="badges">
            <span>20 Models</span><span>HuggingFace Hub</span>
            <span>Resumable</span><span>Auto-organized</span><span>SimplePod Ready</span>
          </div>
        </div>
        """)

        # ── Directory structure preview ────────────────────────────────────────
        gr.HTML("""
        <div class="tree-box">
          <span class="hi">models/</span>  <span class="dim">← base directory (configurable below)</span><br>
          <span class="ok">├── checkpoints/</span>   <span class="dim">ltx-2.3-22b-dev.safetensors  (~41 GB)</span><br>
          <span class="dim">│                 </span><span class="dim">ltx-2.3-22b-distilled-1.1.safetensors  (~41 GB)</span><br>
          <span class="ok">├── components/</span>    <span class="dim">upsampler x2/x1.5  |  distilled-lora  |  temporal-upsampler</span><br>
          <span class="ok">├── text_encoder/</span>  <span class="dim">gemma-3-12b-it-qat-q4_0-unquantized/  (~7 GB)</span><br>
          <span class="ok">└── loras/</span>         <span class="dim">ic-lora-union  |  lipdub  |  LTX-2.3-22b-IC-LoRA-HDR/</span><br>
          <span class="dim">    └── camera/</span>   <span class="dim">dolly-in  |  dolly-out  |  jib-up  |  ...</span>
        </div>
        <div class="warn-box">
          ⚠️  <strong>Note:</strong>
          <code>packages/ltx-core/src/ltx_core/model/</code> contains <strong>Python source code</strong>
          (audio_vae, transformer, video_vae…) — not model weights.
          Model weights (.safetensors) must be stored separately in the <code>models/</code> directory above.
        </div>
        """)

        # ── Settings ──────────────────────────────────────────────────────────
        with gr.Accordion("⚙️  Download Settings", open=True):
            with gr.Row():
                base_dir = gr.Textbox(
                    label="Base Download Directory",
                    value=os.environ.get("LTX_MODEL_DIR", "./models"),
                    placeholder="/workspace/models",
                    info="Subdirectories (checkpoints/, components/, text_encoder/, loras/) are created automatically.",
                    scale=3,
                )
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    value=os.environ.get("HF_TOKEN", ""),
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                    type="password",
                    info="Required for Gemma (gated model). Get at huggingface.co/settings/tokens",
                    scale=2,
                )
            gr.Markdown(
                "> **Gemma** requires accepting the Google Gemma license on HuggingFace before downloading. "
                "Visit [huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized]"
                "(https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) "
                "and click **Agree and access repository**, then provide your token here."
            )

        # ── Quick-select buttons ───────────────────────────────────────────────
        with gr.Row():
            btn_rec  = gr.Button("⭐  Select Recommended", elem_classes="sel-btn")
            btn_all  = gr.Button("☑️  Select All",         elem_classes="sel-btn")
            btn_none = gr.Button("⬜  Deselect All",       elem_classes="sel-btn")
            btn_chk  = gr.Button("🔍  Check Status",       elem_classes="sel-btn")

        # ── Model groups ──────────────────────────────────────────────────────
        gr.HTML(f'<div class="group-hdr">{CATALOG["checkpoints"]["label"]}</div>'
                f'<div class="group-note">{CATALOG["checkpoints"]["note"]}</div>')
        cb_ckpts = gr.CheckboxGroup(label="", choices=_choices("checkpoints"),
                                    value=_defaults("checkpoints"))

        gr.HTML(f'<div class="group-hdr">{CATALOG["components"]["label"]}</div>'
                f'<div class="group-note">{CATALOG["components"]["note"]}</div>')
        cb_comps = gr.CheckboxGroup(label="", choices=_choices("components"),
                                    value=_defaults("components"))

        gr.HTML(f'<div class="group-hdr">{CATALOG["ic_loras"]["label"]}</div>'
                f'<div class="group-note">{CATALOG["ic_loras"]["note"]}</div>')
        cb_icloras = gr.CheckboxGroup(label="", choices=_choices("ic_loras"),
                                      value=_defaults("ic_loras"))

        gr.HTML(f'<div class="group-hdr">{CATALOG["camera_loras"]["label"]}</div>'
                f'<div class="group-note">{CATALOG["camera_loras"]["note"]}</div>')
        cb_cams = gr.CheckboxGroup(label="", choices=_choices("camera_loras"),
                                   value=_defaults("camera_loras"))

        # ── Download button ────────────────────────────────────────────────────
        gr.HTML('<div style="height:12px"></div>')
        dl_btn = gr.Button("⬇️  Download Selected Models",
                           variant="primary", elem_classes="dl-btn")

        log_out = gr.Textbox(
            label="Download Log",
            lines=22, max_lines=22,
            interactive=False, elem_id="log-out",
            placeholder=(
                "Download progress will appear here in real-time...\n\n"
                "• Already-downloaded files are automatically skipped.\n"
                "• Downloads are resumable — safe to restart if interrupted.\n"
                "• Large files (checkpoints ~41 GB) will take a long time on first run."
            ),
        )

        # ── Environment variable reference ────────────────────────────────────
        with gr.Accordion("📋  Environment Variables  (copy these after downloading)", open=True):
            env_md = gr.Markdown(_update_env("./models"))
            gr.Markdown("""
Then launch the generation UI:

```bash
# Copy the export lines above into your shell, then:
bash launch.sh
```

**Optional LoRA paths** (paste into IC-LoRA / HDR / LipDub tabs in app.py):

| LoRA | Path |
|------|------|
| IC-LoRA Union Control | `$MODEL_DIR/loras/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` |
| IC-LoRA LipDub | `$MODEL_DIR/loras/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors` |
| IC-LoRA HDR (repo) | `$MODEL_DIR/loras/LTX-2.3-22b-IC-LoRA-HDR/` |
| Camera LoRAs | `$MODEL_DIR/loras/camera/<filename>.safetensors` |
            """)

        # ── Wire events ───────────────────────────────────────────────────────
        all_cbs = [cb_ckpts, cb_comps, cb_icloras, cb_cams]

        btn_rec.click( _set_recommended, outputs=all_cbs)
        btn_all.click( _set_all,         outputs=all_cbs)
        btn_none.click(_set_none,        outputs=all_cbs)

        btn_chk.click(
            check_status,
            inputs=[base_dir, cb_ckpts, cb_comps, cb_icloras, cb_cams],
            outputs=log_out,
        )

        # Update env var block when directory changes
        base_dir.change(_update_env, inputs=base_dir, outputs=env_md)

        # Persist token + base_dir in localStorage so they survive page refresh.
        # demo.load fires JS first, restores values, then .then() updates env_md.
        demo.load(
            fn=None,
            js="""
            () => {
                const tok = localStorage.getItem('ltx2_hf_token') || '';
                const dir = localStorage.getItem('ltx2_base_dir') || '';
                return [tok, dir];
            }
            """,
            outputs=[hf_token, base_dir],
        ).then(
            fn=_update_env,
            inputs=base_dir,
            outputs=env_md,
        )

        # Save token to localStorage whenever it changes
        hf_token.change(
            fn=None,
            js="(v) => { localStorage.setItem('ltx2_hf_token', v); return v; }",
            inputs=hf_token,
            outputs=hf_token,
        )
        # Save base_dir to localStorage whenever it changes (separate listener
        # from the _update_env listener already wired above)
        base_dir.change(
            fn=None,
            js="(v) => { localStorage.setItem('ltx2_base_dir', v); return v; }",
            inputs=base_dir,
            outputs=base_dir,
        )

        dl_btn.click(
            download_models,
            inputs=[base_dir, hf_token, cb_ckpts, cb_comps, cb_icloras, cb_cams],
            outputs=log_out,
        )

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
    parser = argparse.ArgumentParser(description="LTX-2 Model Downloader")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(max_size=3)
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
