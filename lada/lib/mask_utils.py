# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
from lada.lib import Box, Mask, box_utils
from lada.lib import image_utils


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

def create_blend_mask(crop_mask):
    """Create maximum coverage blend mask that covers all mosaic areas completely."""
    mask = crop_mask.squeeze().float()
    h, w = mask.shape

    # Maximum inclusivity: process any pixel with any mosaic content
    blend = torch.maximum(mask, (mask > 0).float())

    # Multi-scale dilation approach for ultimate edge coverage

    # Phase 1: 13x13 kernel with 5 passes for extensive connectivity
    kernel_size_1 = 13
    kernel_1 = torch.ones(kernel_size_1, kernel_size_1, device=blend.device, dtype=blend.dtype)
    for _ in range(5):
        blend_dilated = F.conv2d(blend.unsqueeze(0).unsqueeze(0),
                               kernel_1.unsqueeze(0).unsqueeze(0),
                               padding=kernel_size_1//2).squeeze()
        # Include areas with very minimal neighboring content (ultra-aggressive)
        new_coverage = (blend_dilated >= 0.01) & (blend == 0)  # Very low threshold
        blend = torch.maximum(blend, new_coverage.float())

    # Additional morphological operations for edge cases
    # Second pass with larger kernel for isolated edge squares
    kernel_size_large = 15
    kernel_large = torch.ones(kernel_size_large, kernel_size_large, device=blend.device, dtype=blend.dtype)

    blend_dilated_large = F.conv2d(blend.unsqueeze(0).unsqueeze(0),
                                 kernel_large.unsqueeze(0).unsqueeze(0),
                                 padding=kernel_size_large//2).squeeze()
    # Catch any remaining isolated mosaic areas with maximum reach
    isolated_coverage = (blend_dilated_large > 0) & (blend == 0)
    blend = torch.maximum(blend, isolated_coverage.float())

    # Phase 3: Final comprehensive connectivity check
    kernel_size_final = 21
    kernel_final = torch.ones(kernel_size_final, kernel_size_final, device=blend.device, dtype=blend.dtype)
    blend_dilated_final = F.conv2d(blend.unsqueeze(0).unsqueeze(0),
                                 kernel_final.unsqueeze(0).unsqueeze(0),
                                 padding=kernel_size_final//2).squeeze()
    # Ensure absolute connectivity across entire mask
    connectivity_coverage = (blend_dilated_final >= 0.001) & (blend == 0)
    blend = torch.maximum(blend, connectivity_coverage.float())

    # Ultimate safety checks - multiple thresholds for absolute coverage
    blend = torch.maximum(blend, (mask > 0).float())
    blend = torch.maximum(blend, (mask > 0.0001).float())  # Catch microscopic traces
    blend = torch.maximum(blend, (blend_dilated >= 0.0001).float())  # Ensure total connectivity

    assert blend.shape == mask.shape
    return blend

def apply_random_mask_extensions(mask: Mask) -> Mask:
    value = np.random.choice([0, 0, 1, 1, 2])
    return extend_mask(mask, value)
