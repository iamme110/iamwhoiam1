import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from lada.lib.image_utils_pt import pad_image, pad_image_by_pad, repad_image, unpad_image, resize, resize_simple, rotate


def test_pad_image():
    """Test pad_image function."""
    img = torch.rand(3, 100, 100)
    padded, pad = pad_image(img, 120, 120)
    assert padded.shape == (3, 120, 120)
    assert pad == (10, 10, 10, 10)  # (10,10,10,10) for symmetric padding


def test_pad_image_by_pad():
    """Test pad_image_by_pad function."""
    img = torch.rand(3, 100, 100)
    pad = (5, 5, 5, 5)
    padded = pad_image_by_pad(img, pad)
    assert padded.shape == (3, 110, 110)


def test_repad_image():
    """Test repad_image function."""
    imgs = [torch.rand(3, 100, 100), torch.rand(3, 100, 100)]
    pads = [(5, 5, 5, 5), (10, 10, 10, 10)]
    repadded = repad_image(imgs, pads)
    assert repadded[0].shape == (3, 100, 100)
    assert repadded[1].shape == (3, 100, 100)


def test_unpad_image():
    """Test unpad_image function."""
    img = torch.rand(3, 120, 120)
    pad = (10, 10, 10, 10)
    unpadded = unpad_image(img, pad)
    assert unpadded.shape == (3, 100, 100)


def test_resize():
    """Test resize function."""
    img = torch.rand(3, 100, 100)
    resized = resize(img, 50)
    # Since 100 > 50, it should resize to (50, something)
    assert resized.shape[0] == 3
    assert resized.shape[1] == 50


def test_resize_simple():
    """Test resize_simple function."""
    img = torch.rand(3, 100, 100)
    resized = resize_simple(img, 50)
    assert resized.shape[0] == 3
    assert resized.shape[1] == 50 or resized.shape[2] == 50


def test_rotate():
    """Test rotate function."""
    img = torch.rand(3, 100, 100)
    rotated = rotate(img, 90)
    assert rotated.shape == (3, 100, 100)  # Assuming square, rotation keeps size


if __name__ == "__main__":
    test_pad_image()
    test_pad_image_by_pad()
    test_repad_image()
    test_unpad_image()
    test_resize()
    test_resize_simple()
    test_rotate()
    print("All tests passed!")
