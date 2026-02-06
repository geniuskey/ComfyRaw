"""
ComfyRaw - Unit tests for comfy_cv.types module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from comfy_cv.types import (
    ensure_float32,
    ensure_uint8,
    ensure_batch,
    validate_image,
    validate_mask,
)


class TestEnsureFloat32:
    """Tests for ensure_float32 function"""

    def test_uint8_to_float32(self):
        """Convert uint8 [0-255] to float32 [0-1]"""
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = ensure_float32(img)

        assert result.dtype == np.float32
        assert result[0, 0, 0] == 0.0
        assert abs(result[0, 0, 1] - 0.5019608) < 0.001
        assert result[0, 0, 2] == 1.0

    def test_float32_passthrough(self):
        """Float32 in [0-1] should pass through unchanged"""
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        result = ensure_float32(img)

        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, img)

    def test_float64_to_float32(self):
        """Float64 should be converted to float32"""
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float64)
        result = ensure_float32(img)

        assert result.dtype == np.float32

    def test_uint16_to_float32(self):
        """uint16 should be converted to float32"""
        img = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
        result = ensure_float32(img)

        assert result.dtype == np.float32
        assert result[0, 0, 0] == 0.0
        assert abs(result[0, 0, 1] - 0.5) < 0.001
        assert result[0, 0, 2] == 1.0


class TestEnsureUint8:
    """Tests for ensure_uint8 function"""

    def test_float32_to_uint8(self):
        """Convert float32 [0-1] to uint8 [0-255]"""
        img = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        result = ensure_uint8(img)

        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 127 or result[0, 0, 1] == 128
        assert result[0, 0, 2] == 255

    def test_uint8_passthrough(self):
        """uint8 should pass through unchanged"""
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        result = ensure_uint8(img)

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, img)

    def test_clipping(self):
        """Values outside [0-1] should be clipped"""
        img = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
        result = ensure_uint8(img)

        assert result.dtype == np.uint8
        assert result[0, 0, 0] == 0
        assert result[0, 0, 2] == 255


class TestEnsureBatch:
    """Tests for ensure_batch function"""

    def test_add_batch_dimension(self):
        """3D array should get batch dimension added"""
        img = np.random.rand(256, 256, 3).astype(np.float32)
        result = ensure_batch(img)

        assert result.ndim == 4
        assert result.shape == (1, 256, 256, 3)

    def test_batch_passthrough(self):
        """4D array should pass through unchanged"""
        img = np.random.rand(2, 256, 256, 3).astype(np.float32)
        result = ensure_batch(img)

        assert result.ndim == 4
        assert result.shape == (2, 256, 256, 3)

    def test_grayscale_batch(self):
        """2D grayscale should become (1, H, W, 1)"""
        img = np.random.rand(256, 256).astype(np.float32)
        result = ensure_batch(img)

        assert result.ndim == 4
        assert result.shape == (1, 256, 256, 1)


class TestValidateImage:
    """Tests for validate_image function"""

    def test_valid_rgb_image(self, sample_image):
        """Valid RGB image should pass validation"""
        # Should not raise
        validate_image(sample_image)

    def test_valid_rgba_image(self, rgba_image):
        """Valid RGBA image should pass validation"""
        validate_image(rgba_image)

    def test_invalid_dimensions(self):
        """Invalid dimensions should raise ValueError"""
        img = np.random.rand(256, 256).astype(np.float32)
        with pytest.raises(ValueError):
            validate_image(img)

    def test_invalid_dtype(self, sample_image):
        """Non-float32 dtype should raise ValueError"""
        img = sample_image.astype(np.float64)
        with pytest.raises(ValueError):
            validate_image(img)


class TestValidateMask:
    """Tests for validate_mask function"""

    def test_valid_mask(self, sample_mask):
        """Valid mask should pass validation"""
        validate_mask(sample_mask)

    def test_invalid_dimensions_4d(self):
        """4D array should raise ValueError"""
        mask = np.random.rand(1, 256, 256, 1).astype(np.float32)
        with pytest.raises(ValueError):
            validate_mask(mask)

    def test_invalid_dimensions_2d(self):
        """2D array should raise ValueError"""
        mask = np.random.rand(256, 256).astype(np.float32)
        with pytest.raises(ValueError):
            validate_mask(mask)
