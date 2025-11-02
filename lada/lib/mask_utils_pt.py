import math

import torch
from torch.nn import functional as torch_functional
from torchvision.transforms import GaussianBlur

from lada.lib import Box, MaskPt
from lada.lib import image_utils_pt


def get_box(mask: MaskPt) -> Box:
    """PyTorch version of get_box. Finds bounding box from mask tensor."""
    mask = mask.squeeze() > 0  # Ensure (H, W) and binary
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    t = torch.where(rows)[0][0].item()
    b = torch.where(rows)[0][-1].item() + 1
    l = torch.where(cols)[0][0].item()
    r = torch.where(cols)[0][-1].item() + 1
    box = t, l, b, r
    return box


def morph(mask: MaskPt, iterations=1) -> MaskPt:
    """PyTorch version of morph. Morphological dilate using convolution with circular kernel."""
    mask = mask.float()  # Convert to float for conv
    if get_mask_area(mask) < 0.01:
        kernel_size = 5
    else:
        kernel_size = 15
    # Create circular kernel to approximate cv2 MORPH_ELLIPSE
    kernel = torch.zeros(kernel_size, kernel_size, dtype=torch.float32, device=mask.device)
    center = kernel_size // 2
    radius = center
    for i in range(kernel_size):
        for j in range(kernel_size):
            if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                kernel[i, j] = 1.0
    for _ in range(iterations):
        mask = torch_functional.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=center) > 0
        mask = mask.squeeze()
    return mask


def dilate_mask(mask: MaskPt, dilatation_size=11, iterations=2) -> MaskPt:
    """PyTorch version of dilate_mask. Dilates mask using convolution."""
    mask = mask.float()  # Convert to float for conv
    if iterations == 0:
        return mask > 0
    kernel = torch.ones(dilatation_size, dilatation_size, dtype=torch.float32, device=mask.device)
    for _ in range(iterations):
        mask = torch_functional.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=dilatation_size // 2) > 0
        mask = mask.squeeze()
    return mask


def extend_mask(mask: MaskPt, value) -> MaskPt:
    """PyTorch version of extend_mask. Extends mask area."""
    if value == 0:
        return mask
    target_size = 256
    # Resize down, morph, resize back, clean boundaries
    resized_down = image_utils_pt.resize(mask.float(), target_size, interpolation='nearest')
    morphed = morph(resized_down)
    resized_back = image_utils_pt.resize(morphed.float(), (int(mask.shape[0]), int(mask.shape[1])), interpolation='nearest')
    cleaned = clean_up_boundaries(resized_back > 0)
    return cleaned


def clean_up_boundaries(mask: MaskPt, kernel_size=19) -> MaskPt:
    """PyTorch version of clean_up_boundaries. Morphological close (dilate then erode)."""
    mask = mask.float()  # Convert to float for conv
    # Close: dilate then erode
    kernel = torch.ones(kernel_size, kernel_size, dtype=torch.float32, device=mask.device)
    # Dilate
    dilated = torch_functional.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size // 2) > 0
    dilated = dilated.squeeze()
    # Erode the dilated
    inverted = 1.0 - dilated.float()
    eroded = 1.0 - (torch_functional.conv2d(inverted.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size // 2) > 0).float()
    eroded = eroded.squeeze()
    return eroded


def fill_holes(mask: MaskPt) -> MaskPt:
    """PyTorch version of fill_holes. Fills holes in mask using flood fill approximation."""
    mask = mask.float()  # Convert to float for conv
    # Simple approximation: dilate and keep original
    # For proper fill, it's complex; this is a basic version
    kernel = torch.ones(3, 3, dtype=torch.float32, device=mask.device)
    dilated = torch_functional.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1) > 0
    filled = dilated.squeeze()
    return filled


def get_mask_area(mask: MaskPt) -> float:
    """PyTorch version of get_mask_area. Calculates mask area ratio."""
    pixels = torch.sum(mask > 0).item()
    return pixels / (mask.shape[0] * mask.shape[1])


def create_blend_mask(crop_mask: MaskPt):
    """PyTorch version of create_blend_mask. Creates blend mask."""
    crop_mask = torch.squeeze(crop_mask) > 0
    h, w = crop_mask.shape
    border_ratio = 0.05
    h_inner, w_inner = int(h * (1.0-border_ratio)), int(w * (1.-border_ratio))
    h_outer, w_outer = h - h_inner, w - w_inner
    border_size = min(h_outer, w_outer)
    if border_size < 5:
        return torch.ones_like(crop_mask, dtype=torch.float32, device=crop_mask.device)
    blur_size = border_size
    blend_mask = torch.ones((h_inner, w_inner), device=crop_mask.device, dtype=torch.float32)
    
    # PyTorch padding: (left, right, top, bottom)
    pad_left = math.floor(w_outer / 2)
    pad_right = math.ceil(w_outer / 2)
    pad_top = math.floor(h_outer / 2)
    pad_bottom = math.ceil(h_outer / 2)
    blend_mask = torch_functional.pad(blend_mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    blend_mask = torch.maximum(crop_mask.float(), blend_mask)
    
    # Apply Gaussian blur using torchvision
    # GaussianBlur expects (C, H, W) format, so add channel dimension
    blend_mask = blend_mask.unsqueeze(0)  # Add channel dimension
    gaussian_blur = GaussianBlur(kernel_size=blur_size if blur_size % 2 == 1 else blur_size + 1, sigma=blur_size/3)
    blend_mask = gaussian_blur(blend_mask)
    blend_mask = blend_mask.squeeze(0)  # Remove channel dimension
    
    assert blend_mask.shape == crop_mask.shape
    return blend_mask
