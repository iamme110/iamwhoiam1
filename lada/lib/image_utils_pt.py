import math

import torch
from torch.nn import functional as torch_functional
from torchvision.transforms.functional import rotate as tv_rotate

from lada.lib import Pad


def pad_image(img: torch.Tensor, max_height: int, max_width: int, mode='zero'):
    """PyTorch version of pad_image. Pads tensor to max_height, max_width."""
    if img.dim() == 3:
        height, width = img.shape[1], img.shape[2]
    elif img.dim() == 2:
        height, width = img.shape[0], img.shape[1]
    else:
        raise ValueError("Tensor must be 2D or 3D")
    if height == max_height and width == max_width:
        return img, (0, 0, 0, 0)
    pad_h = max_height - height
    pad_w = max_width - width
    pad_h_t = math.ceil(pad_h / 2)
    pad_h_b = math.floor(pad_h / 2)
    pad_w_l = math.ceil(pad_w / 2)
    pad_w_r = math.floor(pad_w / 2)
    pad = (pad_h_t, pad_h_b, pad_w_l, pad_w_r)
    padded_image = pad_image_by_pad(img, pad, mode)
    return padded_image, pad


def pad_image_by_pad(img: torch.Tensor, pad: Pad, mode='zero'):
    """PyTorch version of pad_image_by_pad. Pads tensor with given pad."""
    (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
    if mode == 'zero':
        padded_img = torch_functional.pad(img, (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='constant', value=0)
    elif mode == 'reflect':
        padded_img = torch_functional.pad(img, (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='reflect')
    else:
        raise NotImplementedError(f"Mode {mode} not supported")
    return padded_img


def repad_image(imgs: list[torch.Tensor], pads: list[Pad], mode='reflect'):
    """PyTorch version of repad_image. Repads list of tensors."""
    assert len(imgs) == len(pads)
    padded_imgs = []
    for img, pad in zip(imgs, pads):
        (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
        if img.dim() == 3:
            h, w = img.shape[1], img.shape[2]
            cropped = img[:, pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]
        elif img.dim() == 2:
            h, w = img.shape[0], img.shape[1]
            cropped = img[pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]
        else:
            raise ValueError("Tensor must be 2D or 3D")
        if mode == 'zero':
            padded_img = torch_functional.pad(cropped, (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='constant', value=0)
        elif mode == 'reflect':
            padded_img = torch_functional.pad(cropped, (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='reflect')
        else:
            raise NotImplementedError(f"Mode {mode} not supported")
        padded_imgs.append(padded_img)
    return padded_imgs


def unpad_image(img: torch.Tensor, pad: Pad):
    """PyTorch version of unpad_image. Removes padding from tensor."""
    (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
    if img.dim() == 3:
        h, w = img.shape[1], img.shape[2]
        unpadded_img = img[:, pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]
    elif img.dim() == 2:
        h, w = img.shape[0], img.shape[1]
        unpadded_img = img[pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]
    else:
        raise ValueError("Tensor must be 2D or 3D")
    return unpadded_img


def resize(img: torch.Tensor, size: int | tuple[int, int], mode='bilinear', align_corners=False):
    """PyTorch version of resize. Resizes tensor using F.interpolate."""
    if isinstance(size, int):
        if img.dim() == 3:
            h, w = img.shape[1], img.shape[2]
        elif img.dim() == 2:
            h, w = img.shape[0], img.shape[1]
        else:
            raise ValueError("Tensor must be 2D or 3D")
        if max(w, h) == size:
            return img
        if w >= h:
            scale_factor = size / w
            new_h = size
            new_w = math.ceil(h * scale_factor) if scale_factor < 1.0 else math.floor(h * scale_factor)
        else:
            scale_factor = size / h
            new_w = size
            new_h = math.ceil(w * scale_factor) if scale_factor < 1.0 else math.floor(w * scale_factor)
    else:
        new_h, new_w = size
        if img.dim() == 3:
            if img.shape[1] == new_h and img.shape[2] == new_w:
                return img
        elif img.dim() == 2:
            if img.shape[0] == new_h and img.shape[1] == new_w:
                return img
        else:
            raise ValueError("Tensor must be 2D or 3D")
    # Add batch dim if needed
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            resized = torch_functional.interpolate(img, size=(new_h, new_w), mode=mode, align_corners=align_corners)
        else:
            resized = torch_functional.interpolate(img, size=(new_h, new_w), mode=mode)
        resized = resized.squeeze(0).squeeze(0)  # (H, W)
    elif img.dim() == 3:
        img = img.unsqueeze(0)  # (1, C, H, W)
        if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            resized = torch_functional.interpolate(img, size=(new_h, new_w), mode=mode, align_corners=align_corners)
        else:
            resized = torch_functional.interpolate(img, size=(new_h, new_w), mode=mode)
        resized = resized.squeeze(0)  # (C, H, W)
    return resized


def resize_simple(img: torch.Tensor, size: int, mode='bilinear', align_corners=False):
    """PyTorch version of resize_simple. Simple resize keeping aspect ratio."""
    if img.dim() == 3:
        h, w = img.shape[1], img.shape[2]
    elif img.dim() == 2:
        h, w = img.shape[0], img.shape[1]
    else:
        raise ValueError("Tensor must be 2D or 3D")
    if min(w, h) == size:
        return img
    if w >= h:
        new_w = int(size * w / h)
        new_h = size
    else:
        new_w = size
        new_h = int(size * h / w)
    # Resize
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
        if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            resized = torch_functional.interpolate(img, size=(new_h, new_w), mode=mode, align_corners=align_corners)
        else:
            resized = torch_functional.interpolate(img, size=(new_h, new_w), mode=mode)
        resized = resized.squeeze(0).squeeze(0)
    else:  # img.dim() == 3
        img = img.unsqueeze(0)
        if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            resized = torch_functional.interpolate(img, size=(new_h, new_w), mode=mode, align_corners=align_corners)
        else:
            resized = torch_functional.interpolate(img, size=(new_h, new_w), mode=mode)
        resized = resized.squeeze(0)
    return resized


def rotate(img: torch.Tensor, deg: float):
    """PyTorch version of rotate. Rotates tensor using torchvision."""
    # Assume img is (C, H, W) or (H, W)
    if img.dim() == 2:
        img = img.unsqueeze(0)  # Add channel dim
        rotated = tv_rotate(img, deg)
        rotated = rotated.squeeze(0)
    elif img.dim() == 3:
        rotated = tv_rotate(img, deg)
    else:
        raise ValueError("Tensor must be 2D or 3D")
    return rotated
