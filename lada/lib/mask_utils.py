# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
from lada.lib import Box, Mask
from lada.lib import image_utils


def get_box(mask: Mask) -> Box:
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    t, l, b, r = y, x, y+h, x+w
    box = t, l, b, r
    return box

def morph(mask: Mask, iterations=1) -> Mask:
    if get_mask_area(mask) < 0.01:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=iterations)

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
    target_size = 256
    # Dilations are slow when using huge kernels (which we would need for high-res masks). therefore we downscale mask to perform morph operations on much smaller pixel space with smaller kernels
    extended_mask = clean_up_boundaries(image_utils.resize(morph(image_utils.resize(mask, target_size, interpolation=cv2.INTER_NEAREST), iterations=value), mask.shape[:2], interpolation=cv2.INTER_NEAREST)).reshape(mask.shape)
    assert mask.shape == extended_mask.shape
    return extended_mask

def clean_up_boundaries(mask: Mask, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) -> Mask:
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def fill_holes(mask: Mask) -> Mask:
    edited_mask = np.zeros_like(mask, dtype=mask.dtype)
    contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(edited_mask, [cnt], 0, 255, -1)

    return edited_mask

def get_mask_area(mask: Mask) -> float:
    pixels = cv2.countNonZero(mask)
    return pixels / (mask.shape[0] * mask.shape[1])


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
