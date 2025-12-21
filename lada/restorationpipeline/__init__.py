import logging
import os

import torch

from lada import LOG_LEVEL, ModelFiles
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def _find_tensorrt_ep_for_clip(checkpoint_path: str, clip_length: int, fp16: bool) -> str | None:
    from lada.utils.tensorrt_utils import get_compiled_mosaic_restoration_model_path_for_clip

    trt_path = get_compiled_mosaic_restoration_model_path_for_clip(
        checkpoint_path=checkpoint_path,
        clip_length=clip_length,
        fp16=fp16,
    )
    return trt_path if os.path.isfile(trt_path) else None

def _load_small_tensorrt_model(
    mosaic_restoration_checkpoint_path: str,
    mosaic_restoration_config_path: str | None,
    device: torch.device,
    fp16: bool,
    clip_length: int,
):
    from lada.utils.tensorrt_utils import SMALL_TRT_CLIP_LENGTH_TRIGGER

    if clip_length <= SMALL_TRT_CLIP_LENGTH_TRIGGER:
        return None

    assert mosaic_restoration_checkpoint_path.endswith(".pth") or mosaic_restoration_checkpoint_path.endswith(".pt")
    clip10_path = _find_tensorrt_ep_for_clip(
        checkpoint_path=mosaic_restoration_checkpoint_path,
        clip_length=10,
        fp16=fp16,
    )
    if clip10_path is None:
        return None

    from lada.models.basicvsrpp.inference import load_model
    return load_model(mosaic_restoration_config_path, clip10_path, device, fp16)

def load_models(
    device: torch.device,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    mosaic_detection_model_path: str,
    fp16: bool,
    detect_face_mosaics: bool,
    clip_length: int,
):
    if mosaic_restoration_model_name.startswith("deepmosaics"):
        from lada.models.deepmosaics.models import loadmodel
        from lada.restorationpipeline.deepmosaics_mosaic_restorer import DeepmosaicsMosaicRestorer
        _model = loadmodel.video(device, mosaic_restoration_model_path, fp16)
        mosaic_restoration_model = DeepmosaicsMosaicRestorer(_model, device)
        pad_mode = 'reflect'
    elif mosaic_restoration_model_name.startswith("basicvsrpp"):
        from lada.models.basicvsrpp.inference import load_model
        from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
        checkpoint_path = mosaic_restoration_model_path
        weights_path = checkpoint_path
        if device.type == "cuda" and fp16 and (trt_path := _find_tensorrt_ep_for_clip(weights_path, clip_length=clip_length, fp16=fp16)) is not None:
            weights_path = trt_path

        _model = load_model(mosaic_restoration_config_path, weights_path, device, fp16)
        model_clip10 = _load_small_tensorrt_model(checkpoint_path, mosaic_restoration_config_path, device, fp16, clip_length) if weights_path.endswith(".ep") else None
        mosaic_restoration_model = BasicvsrppMosaicRestorer(_model, device, fp16, clip_length, model_clip10=model_clip10)
        pad_mode = 'zero'
    else:
        raise NotImplementedError()
    # setting classes=[0] will consider only detections of class id = 0 (nsfw mosaics) therefore filtering out sfw mosaics (heads, faces)
    if detect_face_mosaics:
        classes = [0]
        detection_model_name = ModelFiles.get_detection_model_by_path(mosaic_detection_model_path)
        if detection_model_name and detection_model_name == "v2":
            logger.info("Mosaic detection model v2 does not support detecting face mosaics. Use detection models v3 or newer. Ignoring...")
    else:
        classes = None
    mosaic_detection_model = Yolo11SegmentationModel(mosaic_detection_model_path, device, classes=classes, conf=0.15, fp16=fp16)
    return mosaic_detection_model, mosaic_restoration_model, pad_mode
