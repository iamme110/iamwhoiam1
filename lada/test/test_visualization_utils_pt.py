import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch

from lada.lib.visualization_utils_pt import overlay_mask, overlay_mask_boundary, draw_box, draw_text, draw_mosaic_detections


def test_overlay_mask():
    """Test overlay_mask function."""
    frame = torch.rand(3, 100, 100)
    mask = torch.rand(100, 100) > 0.5
    overlaid = overlay_mask(frame, mask)
    assert overlaid.shape == frame.shape
    assert overlaid.dtype == frame.dtype


def test_overlay_mask_boundary():
    """Test overlay_mask_boundary function."""
    frame = torch.rand(3, 100, 100)
    mask = torch.zeros(100, 100)
    mask[40:60, 40:60] = 1
    overlaid = overlay_mask_boundary(frame, mask)
    assert overlaid.shape == frame.shape
    assert overlaid.dtype == frame.dtype


def test_draw_box():
    """Test draw_box function."""
    img = torch.rand(3, 100, 100)
    box = (10, 20, 80, 90)  # t, l, b, r
    drawn = draw_box(img, box)
    assert drawn.shape == img.shape
    assert drawn.dtype == img.dtype


def test_draw_text():
    """Test draw_text function."""
    output = torch.rand(3, 100, 100)
    drawn = draw_text("test", (10, 50), output)
    assert drawn.shape == output.shape
    assert drawn.dtype == output.dtype


def test_draw_mosaic_detections():
    """Test draw_mosaic_detections function."""
    # Mock clip
    class MockClip:
        def __init__(self):
            self.id = 1
            self.frame_start = 0
            self.data = [
                (torch.rand(3, 50, 50), torch.rand(50, 50) > 0.5, None, (100, 100), (5, 5, 5, 5)),
                (torch.rand(3, 50, 50), torch.rand(50, 50) > 0.5, None, (100, 100), (5, 5, 5, 5))
            ]
        def __iter__(self):
            return iter(self.data)

    clip = MockClip()
    results = draw_mosaic_detections(clip)
    assert len(results) == 2
    for res in results:
        assert res.shape[0] == 3  # C
        assert res.shape[1] == 50  # H
        assert res.shape[2] == 50  # W


if __name__ == "__main__":
    test_overlay_mask()
    test_overlay_mask_boundary()
    test_draw_box()
    test_draw_text()
    test_draw_mosaic_detections()
    print("All tests passed!")
