# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import threading
from typing import Union
import torch
import torch.nn.functional as torch_functional
import numpy as np
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO
from lada.lib import Image, ImageTorch
from ultralytics.engine.results import Results
class MosaicDetectionResults:
    def __init__(self, orig_img: ImageTorch, path, names, boxes, masks):
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = path
        self.names = names
        self.boxes = [b[:4].int().tolist() for b in boxes]
        self.masks = masks

class PytorchLetterBox:
    def __init__(
            self,
            new_shape: tuple[int, int] = (640, 640),
            stride: int = 32,
            auto:bool = False,
            padding_value: float = 114.0/255.0,
    ):
        self.new_shape = new_shape
        self.stride = stride
        self.auto = auto
        self.padding_value = padding_value

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize and pad a torch.Tensor image with letterboxing.

        Args:
            image (torch.Tensor): Input image tensor of shape (B,C,H,W)

        Returns:
            torch.Tensor: Resized and padded image tensor of shape (B,C,H,W).
        """
        h, w = image.shape[-2:]
        resized_h, resized_w = self.new_shape

        proportion = min(resized_h / h, resized_w / w)
        resized_unpad_w = int(round(w * proportion))
        resized_unpad_h = int(round(h * proportion))

        dw = (resized_w - resized_unpad_w) % self.stride if self.auto else resized_w - resized_unpad_w
        dh = (resized_h - resized_unpad_h) % self.stride if self.auto else resized_h - resized_unpad_h

        if (h, w) != (resized_unpad_h, resized_unpad_w):
            image = torch_functional.interpolate(
                image,
                size=(resized_unpad_h, resized_unpad_w),
                mode="bilinear",
                align_corners=False,
            )

        image = torch_functional.pad(image, (dw // 2, dw - (dw // 2), dh // 2, dh - (dh // 2)), value=self.padding_value)

        return image

class MosaicDetectionModel:
    def __init__(self, model_path: str, device, imgsz=640, half=False, use_torch=False, **kwargs):
        yolo_model = YOLO(model_path)
        self.stride = 32
        self.use_torch = use_torch

        custom = {"conf": 0.25, "save": False, "mode": "predict", "device": device}
        args = {**yolo_model.overrides, **custom, **kwargs}  # highest priority args on the right
        self.args = get_cfg(DEFAULT_CFG, args)
        self.device = torch.device(device)
        self.args.fp16 = half
        self.model = AutoBackend(
            model=yolo_model.model,
            device=self.device,
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=half,
            fuse=True,
            verbose=False,
        )
        self.model.eval()
        self.batch_size:int = self.model.batch if self.model.engine else 8
        self.dtype = torch.float16 if half else torch.float32
        self.is_segmentation_model = (self.model.task if self.model.engine else yolo_model.task) == 'segment'
        imgsz_list = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = ((PytorchLetterBox if use_torch else LetterBox)
                          ((imgsz_list[0], imgsz_list[1]), stride=self.stride, auto=False if self.model.engine else True))
        self._lock = threading.Lock()

    def preprocess(self, imgs: list[Union[Image, ImageTorch]]):
        if self.use_torch:
            batch_imgs = torch.stack([x for x in imgs])\
                .to(self.dtype, memory_format=torch.channels_last)\
                .div_(255.0)
            if len(imgs) != self.batch_size:
                # pad batch
                padding_imgs = torch.zeros(self.batch_size - len(imgs), *batch_imgs.shape[1:], device=batch_imgs.device, dtype=batch_imgs.dtype)
                batch_imgs = torch.cat((batch_imgs, padding_imgs), dim=0)

            return self.letterbox(batch_imgs.permute(0, 3, 1, 2))
        else:
            im = np.stack([self.letterbox(image=x) for x in imgs])
            if len(imgs) != self.batch_size:
                # pad batch
                padding_im = np.zeros((self.batch_size - len(imgs), *im.shape[1:]), dtype=im.dtype)
                im = np.concatenate((im, padding_im), axis=0)
            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
            im = im.to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            return im

    def inference(self, image_batch: torch.Tensor):
        with self._lock:
            with torch.no_grad():
                return self.model(image_batch, augment=False, visualize=False, embed=None)

    def postprocess(self, prediction, img: torch.Tensor, orig_image: list[Union[Image, ImageTorch]]) -> list[Union[Results, MosaicDetectionResults]]:
        protos = prediction[1] if self.model.engine else prediction[1][-1]
        nms_results = nms.non_max_suppression(
            prediction,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
        )
        return [self.construct_result(nms_result, img, orig_img, proto)
                for nms_result, orig_img, proto in zip(nms_results, orig_image, protos)]

    def construct_result(self, nms_result: torch.Tensor, img: torch.Tensor, orig_img: Union[Image, ImageTorch], proto: torch.Tensor) -> Union[Results, MosaicDetectionResults]:
        if not len(nms_result):  # save empty boxes
            masks = None
        else:
            masks = ops.process_mask(proto, nms_result[:, 6:], nms_result[:, :4], img.shape[2:], upsample=True)  # HWC
            nms_result[:, :4] = ops.scale_boxes(img.shape[2:], nms_result[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            nms_result, masks = nms_result[keep], masks[keep]
        return (MosaicDetectionResults if self.use_torch else Results)(orig_img=orig_img, path='', names=self.model.names, boxes=nms_result[:, :6], masks=masks)
