import threading
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as torch_functional
import torchvision.ops as ops

class LetterBox:
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image is torch tensor HWC
        image_tensor = image.permute(2, 0, 1).float()  # CHW float
        height, width = image_tensor.shape[1], image_tensor.shape[2]
        if isinstance(self.new_shape, int):
            new_height, new_width = self.new_shape, self.new_shape
        else:
            new_height, new_width = self.new_shape
        scale_ratio = min(new_height / height, new_width / width)
        if not self.scaleup:
            scale_ratio = min(scale_ratio, 1.0)
        resized_height = int(round(height * scale_ratio))
        resized_width = int(round(width * scale_ratio))
        if resized_height != height or resized_width != width:
            image_tensor = torch_functional.interpolate(image_tensor.unsqueeze(0), size=(resized_height, resized_width), mode='bilinear', align_corners=False).squeeze(0)
        pad_height = new_height - resized_height
        pad_width = new_width - resized_width
        if self.auto:
            pad_height = pad_height % self.stride
            pad_width = pad_width % self.stride
        elif self.scaleFill:
            pad_height = 0
            pad_width = 0
        pad_height /= 2
        pad_width /= 2
        top = int(round(pad_height - 0.1))
        bottom = int(round(pad_height + 0.1))
        left = int(round(pad_width - 0.1))
        right = int(round(pad_width + 0.1))
        # Pad with constant value 114 for all channels
        image_tensor = torch_functional.pad(image_tensor, (left, right, top, bottom), mode='constant', value=114)
        # Back to HWC torch float
        processed_image = image_tensor.permute(1, 2, 0)
        return processed_image

def check_image_size(image_size, stride=32, min_dim=2):
    if isinstance(image_size, int):
        image_size = [image_size, image_size]
    tensor = torch.tensor(image_size, dtype=torch.float)
    tensor = torch.ceil(tensor / stride) * stride
    return tensor.int().tolist()

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, shape[0])  # y1, y2

class Results:
    def __init__(self, orig_img, path, names, boxes, masks):
        self.orig_img = orig_img
        self.path = path
        self.names = names
        self.boxes = boxes
        self.masks = masks
        self.orig_shape = orig_img.shape

class MosaicDetectionModelPT:
    def __init__(self, model_path: str, device: str, imgsz: int = 640, **kwargs) -> None:
        self.device = torch.device(device)
        # Filter kwargs for torch.load supported parameters
        load_kwargs = {k: v for k, v in kwargs.items() if k in ['pickle_module', 'weights_only', 'mmap']}
        self.model = torch.load(model_path, map_location=self.device, **load_kwargs)
        self.model.eval()
        # Handle half precision if specified
        if 'half' in kwargs and kwargs['half']:
            self.model.half()
        # Handle fuse if specified
        if 'fuse' in kwargs and kwargs['fuse']:
            if hasattr(self.model, 'fuse'):
                self.model.fuse()
        # Handle verbose
        self.verbose = kwargs.get('verbose', False)
        self.stride = 32
        self.image_size = check_image_size(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = LetterBox(
            self.image_size,
            auto=True,
            stride=self.stride
        )
        self.is_segmentation_model = True  # Assuming it's a segmentation model
        self.args = type('Args', (), {'conf': 0.25, 'iou': 0.7, 'classes': None, 'agnostic_nms': False, 'max_det': 300})()
        self.names = {0: 'mosaic'}
        self._lock = threading.Lock()

    def preprocess(self, images: List[torch.Tensor]) -> torch.Tensor:
        processed_images = torch.stack([self.letterbox(img) for img in images])  # B H W C
        processed_images = processed_images.permute(0, 3, 1, 2)  # B C H W
        if processed_images.device != self.device:
            processed_images = processed_images.to(self.device)
        processed_images /= 255  # 0 - 255 to 0.0 - 1.0
        return processed_images

    def inference(self, image_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            return self.model(image_batch)

    def postprocess(self, preds: Tuple[torch.Tensor, torch.Tensor], img: torch.Tensor, orig_imgs: List) -> List[Results]:
        pred, proto = preds
        pred = pred[0]
        proto = proto[0]
        pred = self.non_max_suppression(pred, self.args.conf, self.args.iou, self.args.classes, self.args.agnostic_nms, max_det=self.args.max_det)
        pred = pred[0]  # Extract the tensor from the list for single batch
        if len(pred):
            shape = (img.shape[2], img.shape[3])
            masks = self.process_mask(proto, pred[:, 6:], pred[:, :4], shape, upsample=True)
            pred[:, :4] = scale_boxes(shape, pred[:, :4], orig_imgs[0].shape)
            keep = masks.sum(dim=(-2, -1)) > 0
            pred, masks = pred[keep], masks[keep]
        else:
            masks = None
        return [Results(orig_imgs[0], path='', names=self.names, boxes=pred[:, :6], masks=masks)]

    def non_max_suppression(self, prediction: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45, classes: Optional[List[int]] = None, agnostic: bool = False, multi_label: bool = False, labels: Tuple = (), max_det: int = 300) -> List[torch.Tensor]:
        batch_size = prediction.shape[0]  # batch size
        num_classes = prediction.shape[2] - 4 - 32  # number of classes, assuming 32 mask coeffs
        candidates = prediction[..., 4] > conf_thres  # candidates
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * batch_size  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= num_classes > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS
        mask_start_idx = 5 + num_classes  # mask start index
        output = [torch.zeros((0, 6 + 32), device=prediction.device)] * batch_size
        for batch_idx, pred_batch in enumerate(prediction):  # image index, image inference
            pred_batch = pred_batch[candidates[batch_idx]]  # confidence
            if labels and len(labels[batch_idx]):
                label_batch = labels[batch_idx]
                label_tensor = torch.zeros((len(label_batch), num_classes + 5), device=pred_batch.device)
                label_tensor[:, :4] = label_batch[:, 1:5]  # box
                label_tensor[:, 4] = 1.0  # conf
                pred_batch = torch.cat((pred_batch, label_tensor), 0)
            if not pred_batch.shape[0]:
                continue
            pred_batch[:, 5:mask_start_idx] *= pred_batch[:, 4:5]  # conf = obj_conf * cls_conf
            boxes_xyxy = ops.box_convert(pred_batch[:, :4], 'xywh', 'xyxy')
            if multi_label:
                indices, class_indices = (pred_batch[:, 5:mask_start_idx] > conf_thres).nonzero(as_tuple=False).T
                pred_batch = torch.cat((boxes_xyxy[indices], pred_batch[indices, class_indices + 5, None], class_indices[:, None].float()), 1)
            else:  # best class only
                conf_scores, class_indices = pred_batch[:, 5:mask_start_idx].max(1, keepdim=True)
                pred_batch = torch.cat((boxes_xyxy, conf_scores, class_indices.float()), 1)[conf_scores.view(-1) > conf_thres]
            if classes is not None:
                pred_batch = pred_batch[(pred_batch[:, 5:6] == torch.tensor(classes, device=pred_batch.device)).any(1)]
            num_boxes = pred_batch.shape[0]  # number of boxes
            if not num_boxes:  # no boxes
                continue
            pred_batch = pred_batch[pred_batch[:, 4].argsort(descending=True)][:max_nms]  # sort by confidence and remove excess boxes
            class_offset = pred_batch[:, 5:6] * (0 if agnostic else max_wh)  # classes
            nms_boxes, nms_scores = pred_batch[:, :4] + class_offset, pred_batch[:, 4]  # boxes (offset by class), scores
            nms_indices = ops.nms(nms_boxes, nms_scores, iou_thres)  # NMS
            nms_indices = nms_indices[:max_det]  # limit detections
            output[batch_idx] = pred_batch[nms_indices]
        return output

    def process_mask(self, protos: torch.Tensor, masks_in: torch.Tensor, bboxes: torch.Tensor, shape: Tuple[int, int], upsample: bool = False) -> torch.Tensor:
        channels, mask_height, mask_width = protos.shape  # CHW
        img_height, img_width = shape
        masks = (masks_in @ protos.float().view(channels, -1)).view(-1, mask_height, mask_width)  # CHW
        if upsample:
            masks = torch_functional.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        masks = self.crop_mask(masks, bboxes)  # CHW
        return masks.gt_(0.0)

    def crop_mask(self, masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        num_masks, mask_height, mask_width = masks.shape
        left, top, right, bottom = torch.chunk(boxes[:, :, None], 4, 1)  # left shape(n,1,1)
        rows = torch.arange(mask_width, device=masks.device, dtype=left.dtype)[None, None, :]  # rows shape(1,1,w)
        cols = torch.arange(mask_height, device=masks.device, dtype=left.dtype)[None, :, None]  # cols shape(1,h,1)
        return masks * ((rows >= left) * (rows < right) * (cols >= top) * (cols < bottom))
