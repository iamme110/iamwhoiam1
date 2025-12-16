# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import gc
import logging
import os

import torch

from lada import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


def get_compiled_mosaic_restoration_model_path(
    mosaic_restoration_model_path: str,
    clip_length: int,
    fp16: bool,
) -> str:
    precision = "fp16" if fp16 else "fp32"
    output_dir = os.path.dirname(mosaic_restoration_model_path)
    stem = os.path.splitext(os.path.basename(mosaic_restoration_model_path))[0]
    return os.path.join(output_dir, f"{stem}_clip{clip_length}.trt_{precision}.ep")


def compile_mosaic_restoration_model(
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    clip_length: int,
    device: str | torch.device,
    fp16: bool,
    mosaic_restoration_config_path: str | None = None,
) -> str:
    if not mosaic_restoration_model_name.startswith("basicvsrpp"):
        raise ValueError("Only BasicVSR++ restoration models support TensorRT compilation")

    if isinstance(device, str):
        device = torch.device(device)

    output_path = get_compiled_mosaic_restoration_model_path(
        mosaic_restoration_model_path=mosaic_restoration_model_path,
        clip_length=clip_length,
        fp16=fp16,
    )
    if os.path.isfile(output_path):
        return output_path

    from lada.models.basicvsrpp.inference import load_model
    from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer

    model = load_model(mosaic_restoration_config_path, mosaic_restoration_model_path, device, fp16, clip_length)
    restorer = BasicvsrppMosaicRestorer(model, device, fp16, clip_length)
    restorer.compile(output_path=output_path, max_clip_size=clip_length)

    del restorer
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_path
