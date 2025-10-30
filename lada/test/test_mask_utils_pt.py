import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch

from lada.lib import mask_utils
from lada.lib.mask_utils_pt import get_box, morph, dilate_mask, extend_mask, clean_up_boundaries, fill_holes, get_mask_area, create_blend_mask


def test_get_box():
    """Test get_box function."""
    mask = torch.zeros(10, 10)
    mask[2:7, 3:8] = 1
    box = get_box(mask)
    assert box == (2, 3, 7, 8)  # top, left, bottom, right


def test_get_box_comparison():
    """Compare get_box PyTorch and NumPy versions."""
    mask = torch.zeros(10, 10)
    mask[2:7, 3:8] = 1
    np_mask = mask.numpy().astype(np.uint8)
    torch_box = get_box(mask)
    np_box = mask_utils.get_box(np_mask)
    assert torch_box == np_box


def test_morph():
    """Test morph function and verify the result is an approximate ellipse."""
    mask = torch.zeros(10, 10)
    mask[5, 5] = 1
    morphed = morph(mask, iterations=1)
    assert morphed.sum() > 1  # Should expand

    # Verify approximate ellipse: check if the expanded area is roughly circular
    # For a single point dilated with kernel_size=15, it should cover a large area
    # Check the bounding box
    rows = torch.any(morphed, dim=1)
    cols = torch.any(morphed, dim=0)
    height = torch.sum(rows).item()
    width = torch.sum(cols).item()
    # For kernel_size=15, the dilation should cover most of the 10x10 area
    assert height >= 8 and width >= 8  # Approximate check for large expansion
    # Check if center is preserved
    center_row = torch.where(rows)[0][len(torch.where(rows)[0]) // 2].item()
    center_col = torch.where(cols)[0][len(torch.where(cols)[0]) // 2].item()
    assert abs(center_row - 5) <= 1 and abs(center_col - 5) <= 1  # Center near (5,5)


def test_dilate_mask():
    """Test dilate_mask function."""
    mask = torch.zeros(10, 10)
    mask[5, 5] = 1
    dilated = dilate_mask(mask, dilatation_size=3, iterations=1)
    assert dilated.sum() > 1


def test_dilate_mask_comparison():
    """Compare dilate_mask PyTorch and NumPy versions."""
    mask = torch.zeros(10, 10)
    mask[5, 5] = 1
    np_mask = mask.numpy().astype(np.uint8)
    torch_dilated = dilate_mask(mask, dilatation_size=3, iterations=1)
    np_dilated = mask_utils.dilate_mask(np_mask, dilatation_size=3, iterations=1)
    assert np.allclose(torch_dilated.numpy().astype(np.uint8), np_dilated, atol=1e-5)


def test_extend_mask():
    """Test extend_mask function."""
    mask = torch.zeros(50, 50)
    mask[20:30, 20:30] = 1
    extended = extend_mask(mask, 1)
    assert extended.shape == (50, 50)


def test_clean_up_boundaries():
    """Test clean_up_boundaries function."""
    mask = torch.zeros(20, 20)
    mask[5:15, 5:15] = 1
    cleaned = clean_up_boundaries(mask)
    assert cleaned.shape == (20, 20)


def test_clean_up_boundaries_comparison():
    """Compare clean_up_boundaries PyTorch and NumPy versions."""
    mask = torch.zeros(20, 20)
    mask[5:15, 5:15] = 1
    np_mask = mask.numpy().astype(np.uint8)
    torch_cleaned = clean_up_boundaries(mask)
    np_cleaned = mask_utils.clean_up_boundaries(np_mask)
    assert np.allclose(torch_cleaned.numpy().astype(np.uint8), np_cleaned, atol=1e-5)


def test_fill_holes():
    """Test fill_holes function."""
    mask = torch.ones(10, 10)
    mask[3:7, 3:7] = 0
    filled = fill_holes(mask)
    assert filled.sum() > mask.sum()  # Should fill some


def test_get_mask_area():
    """Test get_mask_area function."""
    mask = torch.zeros(10, 10)
    mask[:5, :] = 1
    area = get_mask_area(mask)
    assert abs(area - 0.5) < 0.01


def test_get_mask_area_comparison():
    """Compare get_mask_area PyTorch and NumPy versions."""
    mask = torch.zeros(10, 10)
    mask[:5, :] = 1
    np_mask = mask.numpy()
    torch_area = get_mask_area(mask)
    np_area = mask_utils.get_mask_area(np_mask)
    assert abs(torch_area - np_area) < 1e-5


def test_create_blend_mask():
    """Test create_blend_mask function."""
    mask = torch.zeros(50, 50)
    mask[10:40, 10:40] = 1
    blend = create_blend_mask(mask)
    assert blend.shape == (50, 50)
    assert blend.dtype == torch.float32


def test_create_blend_mask_comparison():
    """Compare create_blend_mask PyTorch and NumPy versions."""
    mask = torch.zeros(50, 50)
    mask[10:40, 10:40] = 1
    np_mask = mask.numpy().astype(np.uint8)
    torch_blend = create_blend_mask(mask)
    np_blend = mask_utils.create_blend_mask(np_mask)
    assert np.allclose(torch_blend.numpy(), np_blend, atol=1e-3)  # Allow some tolerance


if __name__ == "__main__":
    test_get_box()
    test_get_box_comparison()
    test_morph()
    test_dilate_mask()
    test_dilate_mask_comparison()
    test_extend_mask()
    test_clean_up_boundaries()
    test_clean_up_boundaries_comparison()
    test_fill_holes()
    test_get_mask_area()
    test_get_mask_area_comparison()
    test_create_blend_mask()
    test_create_blend_mask_comparison()
    print("All tests passed!")
