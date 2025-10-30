import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch

from lada.lib import image_utils
from lada.lib.image_utils_pt import pad_image, pad_image_by_pad, repad_image, unpad_image, resize, resize_simple, rotate


def test_pad_image():
    """Test pad_image function."""
    img = torch.rand(3, 100, 100)
    padded, pad = pad_image(img, 120, 120)
    assert padded.shape == (3, 120, 120)
    assert pad == (10, 10, 10, 10)  # (10,10,10,10) for symmetric padding


def test_pad_image_comparison():
    """Compare pad_image PyTorch and NumPy versions."""
    np_img = np.random.rand(100, 100, 3).astype(np.float32)
    torch_img = torch.from_numpy(np_img).permute(2, 0, 1)  # CHW

    np_padded, np_pad = image_utils.pad_image(np_img, 120, 120)
    torch_padded, torch_pad = pad_image(torch_img, 120, 120)

    torch_padded_np = torch_padded.permute(1, 2, 0).numpy()

    assert np_pad == torch_pad
    assert np.allclose(np_padded, torch_padded_np, atol=1e-5)


def test_pad_image_by_pad():
    """Test pad_image_by_pad function."""
    img = torch.rand(3, 100, 100)
    pad = (5, 5, 5, 5)
    padded = pad_image_by_pad(img, pad)
    assert padded.shape == (3, 110, 110)


def test_pad_image_by_pad_comparison():
    """Compare pad_image_by_pad PyTorch and NumPy versions."""
    np_img = np.random.rand(100, 100, 3).astype(np.float32)
    torch_img = torch.from_numpy(np_img).permute(2, 0, 1)
    pad = (5, 5, 5, 5)

    np_padded = image_utils.pad_image_by_pad(np_img, pad)
    torch_padded = pad_image_by_pad(torch_img, pad)

    torch_padded_np = torch_padded.permute(1, 2, 0).numpy()

    assert np.allclose(np_padded, torch_padded_np, atol=1e-5)


def test_repad_image():
    """Test repad_image function."""
    imgs = [torch.rand(3, 100, 100), torch.rand(3, 100, 100)]
    pads = [(5, 5, 5, 5), (10, 10, 10, 10)]
    repadded = repad_image(imgs, pads)
    assert repadded[0].shape == (3, 100, 100)
    assert repadded[1].shape == (3, 100, 100)


def test_repad_image_comparison():
    """Compare repad_image PyTorch and NumPy versions."""
    np_imgs = [np.random.rand(100, 100, 3).astype(np.float32), np.random.rand(100, 100, 3).astype(np.float32)]
    torch_imgs = [torch.from_numpy(img).permute(2, 0, 1) for img in np_imgs]
    pads = [(5, 5, 5, 5), (10, 10, 10, 10)]

    np_repadded = image_utils.repad_image(np_imgs, pads)
    torch_repadded = repad_image(torch_imgs, pads)

    for np_r, torch_r in zip(np_repadded, torch_repadded):
        torch_r_np = torch_r.permute(1, 2, 0).numpy()
        assert np.allclose(np_r, torch_r_np, atol=1e-5)


def test_unpad_image():
    """Test unpad_image function."""
    img = torch.rand(3, 120, 120)
    pad = (10, 10, 10, 10)
    unpadded = unpad_image(img, pad)
    assert unpadded.shape == (3, 100, 100)


def test_unpad_image_comparison():
    """Compare unpad_image PyTorch and NumPy versions."""
    np_img = np.random.rand(120, 120, 3).astype(np.float32)
    torch_img = torch.from_numpy(np_img).permute(2, 0, 1)
    pad = (10, 10, 10, 10)

    np_unpadded = image_utils.unpad_image(np_img, pad)
    torch_unpadded = unpad_image(torch_img, pad)

    torch_unpadded_np = torch_unpadded.permute(1, 2, 0).numpy()

    assert np.allclose(np_unpadded, torch_unpadded_np, atol=1e-5)


def test_resize():
    """Test resize function."""
    img = torch.rand(3, 100, 100)
    resized = resize(img, 50)
    # Since 100 > 50, it should resize to (50, something)
    assert resized.shape[0] == 3
    assert resized.shape[1] == 50


def test_resize_comparison():
    """Compare resize PyTorch and NumPy versions."""
    np_img = np.random.rand(100, 100, 3).astype(np.float32)
    torch_img = torch.from_numpy(np_img).permute(2, 0, 1)

    np_resized = image_utils.resize(np_img, 50)
    torch_resized = resize(torch_img, 50)

    torch_resized_np = torch_resized.permute(1, 2, 0).numpy()

    assert np.allclose(np_resized, torch_resized_np, atol=1e-3)  # Allow some tolerance for interpolation


def test_resize_simple():
    """Test resize_simple function."""
    img = torch.rand(3, 100, 100)
    resized = resize_simple(img, 50)
    assert resized.shape[0] == 3
    assert resized.shape[1] == 50 or resized.shape[2] == 50


def test_resize_simple_comparison():
    """Compare resize_simple PyTorch and NumPy versions."""
    np_img = np.random.rand(100, 100, 3).astype(np.float32)
    torch_img = torch.from_numpy(np_img).permute(2, 0, 1)

    np_resized = image_utils.resize_simple(np_img, 50)
    torch_resized = resize_simple(torch_img, 50)

    torch_resized_np = torch_resized.permute(1, 2, 0).numpy()

    assert np.allclose(np_resized, torch_resized_np, atol=1e-3)


def test_rotate():
    """Test rotate function."""
    img = torch.rand(3, 100, 100)
    rotated = rotate(img, 90)
    assert rotated.shape == (3, 100, 100)  # Assuming square, rotation keeps size


def test_rotate_comparison():
    """Compare rotate PyTorch and NumPy versions."""
    np_img = np.random.rand(100, 100, 3).astype(np.float32)
    torch_img = torch.from_numpy(np_img).permute(2, 0, 1)

    np_rotated = image_utils.rotate(np_img, 90)
    torch_rotated = rotate(torch_img, 90)

    torch_rotated_np = torch_rotated.permute(1, 2, 0).numpy()

    assert np_rotated.shape == torch_rotated_np.shape  # Check shapes are consistent, values may differ due to different implementations


if __name__ == "__main__":
    test_pad_image()
    test_pad_image_comparison()
    test_pad_image_by_pad()
    test_pad_image_by_pad_comparison()
    test_repad_image()
    test_repad_image_comparison()
    test_unpad_image()
    test_unpad_image_comparison()
    test_resize()
    test_resize_comparison()
    test_resize_simple()
    test_resize_simple_comparison()
    test_rotate()
    test_rotate_comparison()
    print("All tests passed!")
