"""Regression tests for src.utils.infer_util.resize_foreground.

The repository ships no test harness, and infer_util imports heavy runtime
deps (torch/rembg/imageio) at module top that are unrelated to the pure
numpy/PIL logic exercised here; they are stubbed only when not installed so
the unit under test can run in isolation.
"""
import sys
import types

for _name in ("torch", "rembg", "imageio"):
    try:
        __import__(_name)
    except Exception:
        _m = types.ModuleType(_name)
        if _name == "torch":
            _m.Tensor = type("Tensor", (), {})
        sys.modules[_name] = _m

import numpy as np
import pytest
from PIL import Image

from src.utils.infer_util import resize_foreground


def test_resize_foreground_raises_on_fully_transparent_input():
    # No foreground pixel at all -> np.where index arrays are empty.
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    with pytest.raises(AssertionError):
        resize_foreground(img, 0.85)


def test_resize_foreground_normal_case_still_works():
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[10:20, 10:20, 3] = 255  # a small opaque foreground square
    img = Image.fromarray(arr, "RGBA")
    out = resize_foreground(img, 0.85)
    assert out.mode == "RGBA"
    assert out.size[0] == out.size[1]  # padded to square
