"""
LTX-2 Pipelines: High-level video generation pipelines and utilities.
This package provides ready-to-use pipelines for video generation:
- TI2VidOneStagePipeline: Text/image-to-video in a single stage
- TI2VidTwoStagesPipeline: Two-stage generation with upsampling
- DistilledPipeline: Fast distilled two-stage generation
- ICLoraPipeline: Image/video conditioning with distilled LoRA
- LipDubPipeline: Lip dubbing with IC-LoRA and audio conditioning
- KeyframeInterpolationPipeline: Keyframe-based video interpolation
- RetakePipeline: Regenerate a time region (retake) of an existing video
For more detailed components and utilities, import from specific submodules
like `ltx_pipelines.utils.media_io` or `ltx_pipelines.utils.constants`.
"""

from __future__ import annotations

import importlib

__all__ = [
    "A2VidPipelineTwoStage",
    "DistilledPipeline",
    "ICLoraPipeline",
    "KeyframeInterpolationPipeline",
    "LipDubPipeline",
    "RetakePipeline",
    "TI2VidOneStagePipeline",
    "TI2VidTwoStagesPipeline",
]

_LAZY: dict[str, tuple[str, str]] = {
    "A2VidPipelineTwoStage":        ("ltx_pipelines.a2vid_two_stage",          "A2VidPipelineTwoStage"),
    "DistilledPipeline":            ("ltx_pipelines.distilled",                "DistilledPipeline"),
    "ICLoraPipeline":               ("ltx_pipelines.ic_lora",                  "ICLoraPipeline"),
    "KeyframeInterpolationPipeline":("ltx_pipelines.keyframe_interpolation",   "KeyframeInterpolationPipeline"),
    "LipDubPipeline":               ("ltx_pipelines.lipdub",                   "LipDubPipeline"),
    "RetakePipeline":               ("ltx_pipelines.retake",                   "RetakePipeline"),
    "TI2VidOneStagePipeline":       ("ltx_pipelines.ti2vid_one_stage",         "TI2VidOneStagePipeline"),
    "TI2VidTwoStagesPipeline":      ("ltx_pipelines.ti2vid_two_stages",        "TI2VidTwoStagesPipeline"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'ltx_pipelines' has no attribute {name!r}")
