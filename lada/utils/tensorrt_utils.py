# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import gc
import logging
import os
import sys

import torch

from lada import LOG_LEVEL
from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils.os_utils import get_gpu_vram_gb

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

SMALL_TRT_CLIP_LENGTH = 10
SMALL_TRT_CLIP_LENGTH_TRIGGER = 30

def _approx_max_tensorrt_clip_length(vram_gb: float) -> int:
    if vram_gb < 4:
        return 0
    if vram_gb < 6:
        return 30
    if vram_gb < 8:
        return 60
    if vram_gb < 12:
        return 90
    if vram_gb < 16:
        return 120
    if vram_gb < 24:
        return 180
    if vram_gb < 32:
        return 240
    return 300

def _get_approx_max_tensorrt_clip_length(device: torch.device) -> tuple[float, int]:
    vram_gb = get_gpu_vram_gb(device)
    return vram_gb, _approx_max_tensorrt_clip_length(vram_gb)

def _compile_basicvsrpp_model(model: BasicVSRPlusPlusGan, device: torch.device, dtype: torch.dtype, output_path: str, max_clip_size: int) -> str:
    import psutil
    import torch_tensorrt

    workspace_size = int(psutil.virtual_memory().available * 0.8)
    input = torch.randn(1, max_clip_size, 3, 256, 256, dtype=dtype, device=device)

    with torch_tensorrt.logging.info():
        print(f"Compiling BasicVSR++ model (TensorRT workspace_size={workspace_size / (1024 ** 3):.2f} GB). For large clip length > 100 this can take even few hours.")
        trt_gm = torch_tensorrt.compile(
            model, 
            ir="dynamo", 
            inputs=[input],
            min_block_size=1,
            workspace_size=workspace_size,
            enabled_precisions={dtype},
            use_fp32_acc=False,
            use_explicit_typing=False,
            sparse_weights=False,
            optimization_level=3,
            hardware_compatible=False,
            use_python_runtime=False,
            cache_built_engines=False,
            reuse_cached_engines=False,
            truncate_double=True)

    torch_tensorrt.save(trt_gm, output_path, inputs=[input])
    del trt_gm
    del input
    return output_path

def _get_compiled_mosaic_restoration_model_path(
    mosaic_restoration_model_path: str,
    clip_length: int,
    fp16: bool,
) -> str:
    precision = "fp16" if fp16 else "fp32"
    output_dir = os.path.dirname(mosaic_restoration_model_path)
    stem = os.path.splitext(os.path.basename(mosaic_restoration_model_path))[0]
    return os.path.join(output_dir, f"{stem}_clip{clip_length}.trt_{precision}.ep")

def get_compiled_mosaic_restoration_model_path_for_clip(
    checkpoint_path: str,
    clip_length: int,
    fp16: bool,
) -> str:
    if checkpoint_path.endswith(".ep"):
        raise ValueError("checkpoint_path must be a .pth/.pt path, not a .ep path")
    return _get_compiled_mosaic_restoration_model_path(
        mosaic_restoration_model_path=checkpoint_path,
        clip_length=clip_length,
        fp16=fp16,
    )

def load_ep(checkpoint_path: str, device: torch.device) -> BasicVSRPlusPlusGan:
    logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
    
    import torch_tensorrt
    logger.info(f"Loading TensorRT export from {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        trt_module = torch.export.load(f).module()
        return trt_module.to(device)

def compile_mosaic_restoration_model(
    mosaic_restoration_model_path: str,
    clip_length: int,
    device: str | torch.device,
    fp16: bool,
    mosaic_restoration_config_path: str | None = None,
    interactive: bool = True,
) -> str:

    if isinstance(device, str):
        device = torch.device(device)

    output_path = _get_compiled_mosaic_restoration_model_path(
        mosaic_restoration_model_path=mosaic_restoration_model_path,
        clip_length=clip_length,
        fp16=fp16,
    )
    output_path_small = _get_compiled_mosaic_restoration_model_path(
        mosaic_restoration_model_path=mosaic_restoration_model_path,
        clip_length=SMALL_TRT_CLIP_LENGTH,
        fp16=fp16,
    )
    requested_exists = os.path.isfile(output_path)
    small_exists = os.path.isfile(output_path_small)
    should_use_small_engine = clip_length > SMALL_TRT_CLIP_LENGTH_TRIGGER
    if requested_exists and (small_exists or not should_use_small_engine):
        return output_path

    vram_gb, approx_max_clip_length = _get_approx_max_tensorrt_clip_length(device)
    if approx_max_clip_length == 0:
        print("Skipping compilation due to low VRAM (< 4 GB). Pass --no-compile-mosaic-restoration-model to suppress this message.")
        return output_path if requested_exists else mosaic_restoration_model_path

    if not fp16:
        print("Skipping compilation due to FP32 compilation is not recommended for TensorRT. Consider using FP16 instead to save on VRAM and have faster execution times.")
        return output_path if requested_exists else mosaic_restoration_model_path

    should_compile_requested = not requested_exists
    if clip_length > approx_max_clip_length and should_compile_requested:
        if interactive and sys.stdin.isatty():
            print(
                "\n".join(
                    [
                        f"Requested TensorRT clip length {clip_length}, but GPU VRAM is ~{vram_gb:.1f} GB.",
                        f"Approx safe max is {approx_max_clip_length} frames (rule of thumb: ~2.5 GB per +30 frames).",
                        "",
                        "Large clip lengths can:",
                        "- require significantly more VRAM (compilation may OOM)",
                        "- take much longer to compile",
                        "- on videos with poor mosaic detection the performance may be degraded",
                        "",
                        "Continue compilation anyway? [y/N] ",
                    ]
                ),
                end="",
                flush=True,
            )
            if input().strip().lower() not in {"y", "yes"}:
                should_compile_requested = False
        else:
            print(
                f"Skipping compilation due to low VRAM for requested clip length {clip_length} "
                f"(VRAM ~{vram_gb:.1f} GB, approx safe max {approx_max_clip_length}). "
                "Large clip lengths can require significantly more VRAM, take much longer to compile, and may degrade performance on videos with poor mosaic detection."
            )
            should_compile_requested = False

    from lada.models.basicvsrpp.inference import load_model

    dtype = torch.float16 if fp16 else torch.float32
    should_compile_small = (
        should_use_small_engine
        and (not small_exists)
        and SMALL_TRT_CLIP_LENGTH <= approx_max_clip_length
    )
    if should_compile_small or should_compile_requested:
        model = load_model(mosaic_restoration_config_path, mosaic_restoration_model_path, device, fp16)
        if should_compile_small:
            _compile_basicvsrpp_model(model, device, dtype, output_path_small, SMALL_TRT_CLIP_LENGTH)
        if should_compile_requested and output_path != output_path_small:
            _compile_basicvsrpp_model(model, device, dtype, output_path, clip_length)
        del model

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_path if os.path.isfile(output_path) else mosaic_restoration_model_path
