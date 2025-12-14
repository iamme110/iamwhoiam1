# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from lada.utils import Box, Mask, box_utils
from lada.utils import image_utils

def get_box(mask: Mask) -> Box:
    points = cv2.findNonZero(mask)
    return box_utils.convert_from_opencv(cv2.boundingRect(points))

def morph(mask: Mask, iterations=1, operator=cv2.MORPH_DILATE) -> Mask:
    if get_mask_area(mask) < 0.01:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, operator, kernel, iterations=iterations)

def dilate_mask(mask: Mask, dilatation_size=11, iterations=2):
    if iterations == 0:
        return mask
    element = np.ones((dilatation_size, dilatation_size), np.uint8)
    mask_img = cv2.dilate(mask, element, iterations=iterations).reshape(mask.shape)
    return mask_img

def extend_mask(mask: Mask, value) -> Mask:
    # value between 0 and 3 -> higher values mean more extension of mask area. 0 does not change mask at all
    if value == 0:
        return mask

    # Dilations are slow when using huge kernels (which we would need for high-res masks). therefore we downscale mask to perform morph operations on much smaller pixel space with smaller kernels
    target_size = 256
    extended_mask = image_utils.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    extended_mask = morph(extended_mask, iterations=value, operator=cv2.MORPH_DILATE)
    extended_mask = image_utils.resize(extended_mask, mask.shape[:2], interpolation=cv2.INTER_NEAREST)
    extended_mask = extended_mask.reshape(mask.shape)
    assert mask.shape == extended_mask.shape
    return extended_mask

def clean_mask(mask: Mask, box: Box) -> tuple[Mask, Box]:
    t, l, b, r = box
    # Masks from YOLO prediction extend detection area in some cases. Let's crop
    mask[:t + 1, :, :] = 0
    mask[b:, :, :] = 0
    mask[:, :l + 1, :] = 0
    mask[:, r:, :] = 0

    # Mask from YOLO prediction can sometimes contain additional disconnected (tiny) segments. Keep only the largest
    edited_mask = np.zeros_like(mask, dtype=mask.dtype)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) != 0
    if len(contours) > 1:
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0]
    largest_contour = contours[0]
    cv_box = cv2.boundingRect(largest_contour)
    box = box_utils.convert_from_opencv(cv_box)
    cv2.drawContours(edited_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    return edited_mask, box

def get_mask_area(mask: Mask) -> float:
    pixels = cv2.countNonZero(mask)
    return pixels / (mask.shape[0] * mask.shape[1])

def smooth_mask(mask: Mask, kernel_size: int) -> Mask:
    return cv2.medianBlur(mask, kernel_size).reshape(mask.shape)

def create_blend_mask(crop_mask: torch.Tensor):
    mask = crop_mask.squeeze()
    h, w = mask.shape
    border_ratio = 0.05
    h_inner, w_inner = int(h * (1.0 - border_ratio)), int(w * (1.0 - border_ratio))
    h_outer, w_outer = h - h_inner, w - w_inner
    border_size = min(h_outer, w_outer)
    if border_size < 5:
        return torch.ones_like(mask)
    blur_size = int(border_size)
    if blur_size % 2 == 0:
        blur_size += 1
    inner = torch.ones((h_inner, w_inner), device=mask.device, dtype=mask.dtype)
    pad_top = h_outer // 2
    pad_bottom = h_outer - pad_top
    pad_left = w_outer // 2
    pad_right = w_outer - pad_left
    blend = F.pad(inner, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    mask4 = (mask > 0)
    blend = torch.maximum(mask4, blend)
    kernel = torch.tensor(1.0 / (blur_size**2), device=blend.device, dtype=blend.dtype).expand(1, blur_size, blur_size)
    blend = image_utils.filter2D(blend.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    assert blend.shape == mask.shape
    return blend

def apply_random_mask_extensions(mask: Mask) -> Mask:
    value = np.random.choice([0, 0, 1, 1, 2])
    return extend_mask(mask, value)

def box_to_mask(box: Box, shape, mask_value: int):
    mask = np.zeros((shape[0], shape[1], 1), np.uint8)
    t, l, b, r = box
    mask[t:b + 1, l:r + 1] = mask_value
    return mask


def fix_mask_top_edge(mask: Mask, box: Box, max_extension: int = 15) -> Mask:
    """
    Fixes YOLO mask top edge issue by filling all areas within the bounding box.
    This addresses the known limitation where YOLO masks can be off by a few pixels at the edges,
    with special handling for corners.

    Args:
        mask: Input mask with potential edge gaps
        box: Bounding box (t, l, b, r) that should contain the mask
        max_extension: Maximum number of pixels to extend/fill

    Returns:
        Mask with edges filled to match the bounding box
    """
    t, l, b, r = box

    # Create a copy to avoid modifying the original
    fixed_mask = mask.copy()

    # Step 1: Fill gaps at the top edge by extending downward from existing content
    for row in range(t, min(b + 1, mask.shape[0])):
        if np.any(fixed_mask[row, l:r+1, :] > 0):
            # This row has content, use it to fill upward
            for fill_row in range(max(t, row - max_extension), row):
                if not np.any(fixed_mask[fill_row, l:r+1, :] > 0):
                    # Copy content from the current row to fill the gap
                    fixed_mask[fill_row, l:r+1, :] = fixed_mask[row, l:r+1, :]
            break

    # Step 2: Enhanced morphological filling with corner-specific handling
    bbox_region = fixed_mask[t:b+1, l:r+1, :]

    if np.sum(bbox_region > 0) > 0:
        # Use morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        filled_region = cv2.morphologyEx(bbox_region.squeeze(), cv2.MORPH_CLOSE, kernel)

        # Step 3: Corner-specific filling - be more aggressive for corners
        # Check if corners are empty and within bounding box
        h, w = filled_region.shape
        corner_fill_distance = min(max_extension, min(h, w) // 4)  # Adaptive corner distance

        # Create distance transform for intelligent filling
        dist_transform = cv2.distanceTransform((filled_region > 0).astype(np.uint8), cv2.DIST_L2, 3)

        # Fill areas close to existing content
        fill_mask = (dist_transform <= max_extension) & (filled_region == 0)

        # Additional corner filling - more aggressive for top corners
        if t > 0:  # If we're not at the very top of the image
            # Check top-left and top-right corners specifically
            top_region = filled_region[:corner_fill_distance, :]
            if np.sum(top_region > 0) > 0:  # If there's some content in the top region
                # Fill top corners more aggressively
                top_dist = cv2.distanceTransform((top_region > 0).astype(np.uint8), cv2.DIST_L2, 3)
                top_fill_mask = (top_dist <= corner_fill_distance * 1.5) & (top_region == 0)
                top_region[top_fill_mask] = 255

                # Special handling for left top corner - even more aggressive
                left_corner_region = top_region[:, :corner_fill_distance]
                if np.sum(left_corner_region > 0) > 0:
                    left_dist = cv2.distanceTransform((left_corner_region > 0).astype(np.uint8), cv2.DIST_L2, 3)
                    left_fill_mask = (left_dist <= corner_fill_distance * 2.0) & (left_corner_region == 0)
                    left_corner_region[left_fill_mask] = 255

        # Apply morphological dilation for final cleanup
        filled_region = cv2.morphologyEx(filled_region, cv2.MORPH_DILATE, kernel, iterations=1)

        # Smarter safety check: allow filling up to 3x the original content or 60% of bbox, whichever is smaller
        original_bbox_content = np.sum(bbox_region > 0)
        bbox_area = (b - t + 1) * (r - l + 1)
        max_allowed_fill = min(original_bbox_content * 3, bbox_area * 0.6)
        filled_pixels = np.sum(filled_region > 0)
        if filled_pixels <= max_allowed_fill:
            fixed_mask[t:b+1, l:r+1, :] = filled_region[:, :, np.newaxis]

    return fixed_mask