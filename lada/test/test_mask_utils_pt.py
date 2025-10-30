import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from lada.lib.mask_utils_pt import get_box, morph, dilate_mask, extend_mask, clean_up_boundaries, fill_holes, get_mask_area, create_blend_mask


def test_get_box():
    """Test get_box function."""
    mask = torch.zeros(10, 10)
    mask[2:7, 3:8] = 1
    box = get_box(mask)
    assert box == (2, 3, 6, 7)  # top, left, bottom, right


def test_morph():
    """Test morph function."""
    mask = torch.zeros(10, 10)
    mask[5, 5] = 1
    morphed = morph(mask, iterations=1)
    assert morphed.sum() > 1  # Should expand


def test_dilate_mask():
    """Test dilate_mask function."""
    mask = torch.zeros(10, 10)
    mask[5, 5] = 1
    dilated = dilate_mask(mask, dilatation_size=3, iterations=1)
    assert dilated.sum() > 1


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


def test_create_blend_mask():
    """Test create_blend_mask function."""
    mask = torch.zeros(50, 50)
    mask[10:40, 10:40] = 1
    blend = create_blend_mask(mask)
    assert blend.shape == (50, 50)
    assert blend.dtype == torch.float32


if __name__ == "__main__":
    test_get_box()
    test_morph()
    test_dilate_mask()
    test_extend_mask()
    test_clean_up_boundaries()
    test_fill_holes()
    test_get_mask_area()
    test_create_blend_mask()
    print("All tests passed!")
