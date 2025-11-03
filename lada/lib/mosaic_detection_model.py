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
from lada.lib import Image, ImagePt
from ultralytics.engine.results import Results
class MosaicDetectionResults:
    def __init__(self, orig_img: ImagePt, path, names, boxes, masks):
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
            scaleup: bool = True,
            stride: int = 32,
            padding_value: float = 0.447,  # 114/255 for normalized images
    ):
        self.new_shape = new_shape
        self.scaleup = scaleup
        self.stride = stride
        self.padding_value = padding_value

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize and pad a torch.Tensor image with letterboxing.

        Args:
            image (torch.Tensor): Input image tensor of shape (H, W, C)

        Returns:
            torch.Tensor: Resized and padded image tensor of shape (H, W, C).
        """
        # Transpose from HWC to CHW for processing
        image = image.permute(2, 0, 1)
        c, h, w = image.shape

        # Calculate scale ratio
        r = min(self.new_shape[0] / h, self.new_shape[1] / w)
        if not self.scaleup:
            r = min(r, 1.0)

        # Calculate new unpadded size
        new_unpad = (int(round(w * r)), int(round(h * r)))

        # Resize if needed
        if (h, w) != (int(round(h * r)), int(round(w * r))):
            is_byte = image.dtype == torch.uint8
            if is_byte:
                image = image.float() / 255.0

            image = torch_functional.interpolate(
                image.unsqueeze(0),
                size=(int(round(h * r)), int(round(w * r))),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            if is_byte:
                image = (image * 255).byte()

        # Calculate padding (auto mode with stride alignment)
        dw = self.new_shape[1] - new_unpad[0]
        dh = self.new_shape[0] - new_unpad[1]

        # Apply stride alignment (auto mode)
        dw = dw % self.stride
        dh = dh % self.stride

        # Center the padding
        dw /= 2
        dh /= 2

        # Calculate padding values
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        # Apply padding
        if top > 0 or bottom > 0 or left > 0 or right > 0:
            image = torch_functional.pad(image, (left, right, top, bottom), value=self.padding_value)

        # Transpose back from CHW to HWC
        return image.permute(1, 2, 0)

class MosaicDetectionModel:
    def __init__(self, model_path: str, device, imgsz=640, use_torch=False, **kwargs):
        yolo_model = YOLO(model_path)
        assert yolo_model.task == 'segment'
        self.stride = 32
        self.use_torch = use_torch
        imgsz_list = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = PytorchLetterBox(
            (imgsz_list[0], imgsz_list[1]),
            scaleup=True,
            stride=self.stride,
            padding_value=0.447
        ) if use_torch \
        else LetterBox(
            (imgsz_list[0], imgsz_list[1]),
            auto=True,
            stride=self.stride
        )

        custom = {"conf": 0.25, "batch": 4, "save": False, "mode": "predict", "device": device}
        args = {**yolo_model.overrides, **custom, **kwargs}  # highest priority args on the right
        self.args = get_cfg(DEFAULT_CFG, args)
        self.device = torch.device(device)

        self.model = AutoBackend(
            model=yolo_model.model,
            device=self.device,
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=False,
        )
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, imgsz_list[0], imgsz_list[1]))

        self.is_segmentation_model = yolo_model.task == 'segment'
        self._lock = threading.Lock()

    def preprocess(self, imgs: list[Union[Image, ImagePt]]):
        if self.use_torch:
            processed_images = torch.stack([self.letterbox(img) for img in imgs])  # B H W C
            return processed_images.to(self.device, torch.float32).div_(255.0).permute(0, 3, 1, 2)  # permute to B C H W
        else:
            im = np.stack([self.letterbox(image=x) for x in imgs])
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

    def postprocess(self, inference_results, img: torch.Tensor, orig_image: list[Union[Image, ImagePt]]) -> list[Union[Results, MosaicDetectionResults]]:
        protos = inference_results[1][-1]
        inference_results = nms.non_max_suppression(
            inference_results,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
        )
        return [self.construct_result(inference_result, img, orig_img, proto)
                for inference_result, orig_img, proto in zip(inference_results, orig_image, protos)]
    
    def construct_result(self, inference_result: torch.Tensor, img: torch.Tensor, orig_img: Union[Image, ImagePt], proto: torch.Tensor) -> Union[Results, MosaicDetectionResults]:
        if not len(inference_result):  # save empty boxes
            masks = None
        else:
            masks = ops.process_mask(proto, inference_result[:, 6:], inference_result[:, :4], img.shape[2:], upsample=True)  # HWC
            inference_result[:, :4] = ops.scale_boxes(img.shape[2:], inference_result[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            inference_result, masks = inference_result[keep], masks[keep]
        return (MosaicDetectionResults if self.use_torch else Results)(orig_img=orig_img, path='', names=self.model.names, boxes=inference_result[:, :6], masks=masks)
