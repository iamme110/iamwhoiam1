import torch
import torchvision.utils as tv_utils
import cv2
import numpy as np

from lada.lib import Box
from lada.lib.image_utils import unpad_image
from lada.lib.image_utils_pt import resize, pad_image_by_pad
from lada.lib.mask_utils_pt import dilate_mask
from lada.lib.mosaic_detector import Clip


def overlay_mask(frame: torch.Tensor, mask: torch.Tensor):
    """PyTorch version of overlay_mask. Overlay mask on frame using torchvision."""
    mask = mask > 0  # Ensure bool
    frame_uint8 = (frame * 255).to(torch.uint8)
    overlapped = tv_utils.draw_segmentation_masks(frame_uint8, mask, alpha=0.1, colors=[(255, 255, 255)])
    return overlapped.float() / 255.0


def overlay_mask_boundary(frame: torch.Tensor, mask: torch.Tensor, color=(0, 255, 0)):
    """PyTorch version of overlay_mask_boundary. Draw mask boundary using torchvision."""
    mask = mask > 0  # Ensure bool
    # Calculate boundary mask: dilated - mask
    dilated = dilate_mask(mask, dilatation_size=3, iterations=1)
    boundary_mask = dilated & ~mask  # outer boundary
    frame_uint8 = (frame * 255).to(torch.uint8)
    overlapped = tv_utils.draw_segmentation_masks(frame_uint8, boundary_mask, alpha=1.0, colors=[color])
    return overlapped.float() / 255.0


def draw_box(img: torch.Tensor, box: Box, color=(255, 0, 0), thickness=2):
    """PyTorch version of draw_box. Draw bounding box using torchvision."""
    boxes = torch.tensor([[box[1], box[0], box[3], box[2]]], dtype=torch.float32)  # (x1, y1, x2, y2)
    colors = [color]
    img_uint8 = (img * 255).to(torch.uint8)
    drawn = tv_utils.draw_bounding_boxes(img_uint8, boxes, colors=colors, width=thickness)
    return drawn.float() / 255.0


def draw_text(text: str, position: tuple[int, int], output: torch.Tensor, font_scale=0.5):
    """PyTorch version of draw_text. Draw text using cv2."""
    # Convert to numpy HWC
    output_np = output.permute(1, 2, 0).numpy() * 255
    output_np = output_np.astype(np.uint8)
    output_np = np.ascontiguousarray(output_np)
    cv2.putText(output_np, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    # Convert back to tensor CHW
    output_tensor = torch.from_numpy(output_np).permute(2, 0, 1).float() / 255
    return output_tensor


def draw_mosaic_detections(clip:Clip, border_color=(255, 0, 255)) -> list[torch.Tensor]:
    """PyTorch version of draw_mosaic_detections. Process clip with PyTorch operations."""
    mosaic_detection_images = []
    box_border_thickness = 2
    border_thickness_half = box_border_thickness // 2
    for (cropped_img, cropped_mask, _, orig_crop_shape, pad_after_resize) in clip:
        # Assume cropped_img is torch.Tensor (C, H, W), cropped_mask is torch.Tensor (H, W)
        mosaic_detection_img = cropped_img.clone()

        # Draw text
        mosaic_detection_img = draw_text(f"c:{clip.id},f_start:{clip.frame_start}", (25, cropped_img.shape[2] // 2), mosaic_detection_img)

        # Unpad, resize, etc.
        mosaic_detection_img = unpad_image(mosaic_detection_img, pad_after_resize)
        shape_before_resize = mosaic_detection_img.shape[1:]  # (H, W)
        mosaic_detection_img = resize(mosaic_detection_img, orig_crop_shape[:2])

        t, l, b, r = 0, 0, mosaic_detection_img.shape[1] - 1, mosaic_detection_img.shape[2] - 1
        border_box = (t + border_thickness_half, l + border_thickness_half, b - border_thickness_half, r - border_thickness_half)

        mosaic_detection_img = draw_box(mosaic_detection_img, border_box, color=border_color, thickness=box_border_thickness)

        mosaic_detection_img = resize(mosaic_detection_img, shape_before_resize)
        mosaic_detection_img = pad_image_by_pad(mosaic_detection_img, pad_after_resize)

        assert mosaic_detection_img.shape == cropped_img.shape, "shapes of mosaic detection img and cropped img must match"

        mosaic_detection_img = overlay_mask_boundary(mosaic_detection_img, cropped_mask)

        mosaic_detection_images.append(mosaic_detection_img)
    return mosaic_detection_images
