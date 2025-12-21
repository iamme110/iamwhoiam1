import logging
import os

import torch

from lada import LOG_LEVEL, ModelFiles
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def _load_small_tensorrt_model(
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    device: torch.device,
    fp16: bool,
    clip_length: int,
):
    from lada.utils.tensorrt_utils import SMALL_TRT_CLIP_LENGTH_TRIGGER, get_compiled_mosaic_restoration_model_path_for_clip

    if clip_length <= SMALL_TRT_CLIP_LENGTH_TRIGGER:
        return None

    if mosaic_restoration_model_path.endswith(".ep"):
        if "_clip" not in mosaic_restoration_model_path:
            return None
        prefix, rest = mosaic_restoration_model_path.rsplit("_clip", 1)
        idx = rest.find(".trt_")
        if idx == -1:
            return None
        clip10_path = f"{prefix}_clip10{rest[idx:]}"
        if clip10_path == mosaic_restoration_model_path:
            return None
    else:
        clip10_path = get_compiled_mosaic_restoration_model_path_for_clip(
            checkpoint_path=mosaic_restoration_model_path,
            clip_length=10,
            fp16=fp16,
        )
    if not os.path.isfile(clip10_path):
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
        _model = load_model(mosaic_restoration_config_path, mosaic_restoration_model_path, device, fp16)
        model_clip10 = _load_small_tensorrt_model(
            mosaic_restoration_model_path=mosaic_restoration_model_path,
            mosaic_restoration_config_path=mosaic_restoration_config_path,
            device=device,
            fp16=fp16,
            clip_length=clip_length,
        )

        mosaic_restoration_model = BasicvsrppMosaicRestorer(
            _model,
            device,
            fp16,
            clip_length,
            model_clip10=model_clip10,
        )
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
