import math

import cv2
import torch
from torch.nn import functional as torch_functional
from torchvision.transforms.functional import rotate as tv_rotate

from lada.lib import Pad


def pad_image(img: torch.Tensor, max_height: int, max_width: int, mode='zero'):
    """PyTorch version of pad_image. Pads tensor to max_height, max_width."""
    # For HWC format, height and width are at indices 0 and 1
    height, width = img.shape[0:2]
    if height == max_height and width == max_width:
        return img, [0, 0, 0, 0]
    
    pad_h = max_height - height
    pad_w = max_width - width
    pad_h_t = math.ceil(pad_h / 2)
    pad_h_b = math.floor(pad_h / 2)
    pad_w_l = math.ceil(pad_w / 2)
    pad_w_r = math.floor(pad_w / 2)
    
    pad = [pad_h_t, pad_h_b, pad_w_l, pad_w_r]
    
    padded_image = pad_image_by_pad(img, pad, mode)
    # For HWC format, check height and width at indices 0 and 1
    assert padded_image.shape[0:2] == (max_height, max_width)
    return padded_image, pad


def pad_image_by_pad(img: torch.Tensor, pad: Pad, mode='zero'):
    """PyTorch version of pad_image_by_pad. Pads tensor with given pad."""
    pad_h_t, pad_h_b, pad_w_l, pad_w_r = pad
    
    if img.ndim == 3:
        # For 3D tensor (H, W, C) - need to permute to (C, H, W) for torch_functional.pad
        img_chw = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        img_bchw = img_chw.unsqueeze(0)  # (1, C, H, W)
        if mode == 'zero':
            # torch_functional.pad format: (pad_left, pad_right, pad_top, pad_bottom)
            padded_img = torch_functional.pad(img_bchw, (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='constant', value=0)
        elif mode == 'reflect':
            padded_img = torch_functional.pad(img_bchw, (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='reflect')
        else:
            raise NotImplementedError()
        # Remove batch dimension and permute back to (H, W, C)
        padded_img = padded_img.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    else:
        raise NotImplementedError("pad_image_by_pad currently only supports 3D tensors (H, W, C)")
    
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


def resize(img: torch.Tensor, size: int | tuple[int, int], interpolation='bilinear', align_corners=False):
    """
    Resize a torch.Tensor image using PyTorch's torch_functional.interpolate.
    
    Args:
        img (torch.Tensor): Input tensor of shape (H, W, C) or (B, H, W, C)
        size (int|tuple[int, int]): Target size. If int, resize keeping aspect ratio 
                                   with max dimension equal to size. If tuple, exact (H, W) size.
        interpolation (str|int): Interpolation mode. Can be PyTorch mode string
                               ('bilinear', 'bicubic', 'nearest', 'area') or OpenCV constant
                               (cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST)
        align_corners (bool): Align corners parameter for 'bilinear' and 'bicubic' modes.
    
    Returns:
        torch.Tensor: Resized tensor in (H, W, C) or (B, H, W, C) format
    """
    # Map OpenCV interpolation constants to PyTorch interpolation modes
    if isinstance(interpolation, int):
        cv2_to_torch_interp = {
            cv2.INTER_LINEAR: 'bilinear',
            cv2.INTER_CUBIC: 'bicubic', 
            cv2.INTER_NEAREST: 'nearest',
            cv2.INTER_AREA: 'area'
        }
        interpolation = cv2_to_torch_interp.get(interpolation, 'bilinear')
    
    # Store original dtype for conversion back
    original_dtype = img.dtype
    
    # Handle special dtypes that torch_functional.interpolate doesn't support
    convert_back_to_bool = False
    convert_back_to_int = False
    
    if img.dtype == torch.bool:
        # For bool, convert to uint8 for nearest interpolation
        img = img.to(torch.uint8)
        convert_back_to_bool = True
        interpolation = 'nearest'
    elif img.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        # For integer types, use nearest neighbor without conversion
        convert_back_to_int = True
        interpolation = 'nearest'
    
    # Get dimensions
    original_shape = img.shape
    if len(original_shape) == 3:  # (H, W, C)
        h, w, c = img.shape
    else:  # (B, H, W, C)
        b, h, w, c = img.shape
    
    # Calculate new size
    if type(size) == int:
        # Keep aspect ratio, resize so max dimension equals size
        if max(w, h) == size:
            # No resize needed, convert back to original dtype if needed
            if convert_back_to_bool:
                img = img.to(torch.bool)
            return img
            
        if w >= h:
            scale_factor = size / w
            new_w = size
            new_h = math.ceil(h * scale_factor) if scale_factor < 1.0 else math.floor(h * scale_factor)
        else:
            scale_factor = size / h
            new_h = size
            new_w = math.ceil(w * scale_factor) if scale_factor < 1.0 else math.floor(w * scale_factor)
        new_size = (new_h, new_w)
    else:
        # Exact size
        new_h, new_w = size
        if (h, w) == (new_h, new_w):
            # No resize needed, convert back to original dtype if needed
            if convert_back_to_bool:
                img = img.to(torch.bool)
            return img
        new_size = (new_h, new_w)
    
    # Handle batch dimension and format conversion for interpolation
    if len(original_shape) == 3:  # (H, W, C)
        # Convert to (C, H, W) then add batch dimension: (1, C, H, W)
        img = img.permute(2, 0, 1).unsqueeze(0)
        squeeze_output = True
    else:  # (B, H, W, C)
        # Convert to (B, C, H, W)
        img = img.permute(0, 3, 1, 2)
        squeeze_output = False
    
    # Use torch_functional.interpolate for resizing
    resized_img = torch_functional.interpolate(img, size=new_size, mode=interpolation, align_corners=align_corners if interpolation in ['bilinear', 'bicubic'] else None)
    
    # Convert back to original dtype if needed
    if convert_back_to_bool:
        # Convert back to bool
        resized_img = resized_img.to(torch.bool)
    
    # Convert back to original format (H, W, C) or (B, H, W, C)
    if squeeze_output:
        resized_img = resized_img.squeeze(0).permute(1, 2, 0)  # (1, C, H, W) -> (H, W, C)
    else:
        resized_img = resized_img.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    
    # Verify output size
    if type(size) == int:
        assert size == max(resized_img.shape[-3:-1]), f"Expected max dimension {size}, got {max(resized_img.shape[-3:-1])}"
    else:
        assert resized_img.shape[-3:-1] == torch.Size(size), f"Expected size {size}, got {resized_img.shape[-3:-1]}"
    
    return resized_img


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
