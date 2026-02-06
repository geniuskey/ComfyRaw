"""
ComfyRaw - Unit tests for basic nodes in nodes.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from PIL import Image

from nodes import (
    LoadImage,
    SaveImage,
    PreviewImage,
    ImageScale,
    ImageScaleBy,
    ImageInvert,
    ImageBatch,
    EmptyImage,
    ImageFlip,
    ImageRotate,
    ImageCrop,
)


class TestLoadImage:
    """Tests for LoadImage node"""

    def test_input_types(self):
        """Check INPUT_TYPES structure"""
        input_types = LoadImage.INPUT_TYPES()

        assert "required" in input_types
        assert "image" in input_types["required"]

    def test_load_image(self, input_dir, monkeypatch):
        """Load an image file"""
        import folder_paths
        monkeypatch.setattr(folder_paths, "get_input_directory", lambda: input_dir)

        node = LoadImage()
        result = node.load_image("test_image.png")

        assert len(result) == 2  # (image, mask)
        image, mask = result

        assert image.ndim == 4
        assert image.dtype == np.float32
        assert image.shape[0] == 1
        assert image.shape[3] == 3

    def test_load_image_with_alpha(self, input_dir, monkeypatch):
        """Load an image with alpha channel"""
        import folder_paths
        monkeypatch.setattr(folder_paths, "get_input_directory", lambda: input_dir)

        node = LoadImage()
        result = node.load_image("test_image_alpha.png")

        image, mask = result

        assert image.shape[3] == 3  # RGB only
        assert mask.ndim == 3  # (B, H, W)


class TestImageScale:
    """Tests for ImageScale node"""

    def test_input_types(self):
        """Check INPUT_TYPES structure"""
        input_types = ImageScale.INPUT_TYPES()

        assert "required" in input_types
        assert "image" in input_types["required"]
        assert "width" in input_types["required"]
        assert "height" in input_types["required"]

    def test_upscale(self, sample_image_tensor):
        """Scale image to larger size"""
        node = ImageScale()
        result = node.upscale(sample_image_tensor, "bilinear", 512, 512, "disabled")

        image = result[0]
        assert image.shape == (1, 512, 512, 3)

    def test_downscale(self, sample_image_tensor):
        """Scale image to smaller size"""
        node = ImageScale()
        result = node.upscale(sample_image_tensor, "bilinear", 64, 64, "disabled")

        image = result[0]
        assert image.shape == (1, 64, 64, 3)

    def test_crop_mode(self, sample_image_tensor):
        """Scale with center crop"""
        node = ImageScale()
        result = node.upscale(sample_image_tensor, "bilinear", 128, 64, "center")

        image = result[0]
        assert image.shape == (1, 64, 128, 3)


class TestImageScaleBy:
    """Tests for ImageScaleBy node"""

    def test_scale_up(self, sample_image_tensor):
        """Scale up by factor"""
        node = ImageScaleBy()
        result = node.upscale(sample_image_tensor, "bilinear", 2.0)

        image = result[0]
        assert image.shape == (1, 512, 512, 3)

    def test_scale_down(self, sample_image_tensor):
        """Scale down by factor"""
        node = ImageScaleBy()
        result = node.upscale(sample_image_tensor, "bilinear", 0.5)

        image = result[0]
        assert image.shape == (1, 128, 128, 3)


class TestImageInvert:
    """Tests for ImageInvert node"""

    def test_invert_black(self):
        """Invert black image to white"""
        node = ImageInvert()
        black = np.zeros((1, 64, 64, 3), dtype=np.float32)
        result = node.invert(black)

        image = result[0]
        np.testing.assert_array_almost_equal(image, np.ones_like(image))

    def test_invert_white(self):
        """Invert white image to black"""
        node = ImageInvert()
        white = np.ones((1, 64, 64, 3), dtype=np.float32)
        result = node.invert(white)

        image = result[0]
        np.testing.assert_array_almost_equal(image, np.zeros_like(image))


class TestImageBatch:
    """Tests for ImageBatch node"""

    def test_batch_two_images(self, sample_image_tensor):
        """Batch two images together"""
        node = ImageBatch()
        image1 = sample_image_tensor
        image2 = np.random.rand(1, 256, 256, 3).astype(np.float32)

        result = node.batch(image1, image2)
        batched = result[0]

        assert batched.shape == (2, 256, 256, 3)

    def test_batch_different_sizes(self):
        """Batch images with different sizes"""
        node = ImageBatch()
        image1 = np.random.rand(1, 128, 128, 3).astype(np.float32)
        image2 = np.random.rand(1, 256, 256, 3).astype(np.float32)

        result = node.batch(image1, image2)
        batched = result[0]

        # Second image should be resized to match first
        assert batched.shape[1:] == image1.shape[1:]


class TestEmptyImage:
    """Tests for EmptyImage node"""

    def test_create_black(self):
        """Create black empty image"""
        node = EmptyImage()
        result = node.generate(512, 512, 1, 0)

        image = result[0]
        assert image.shape == (1, 512, 512, 3)
        assert np.allclose(image, 0)

    def test_create_white(self):
        """Create white empty image"""
        node = EmptyImage()
        result = node.generate(512, 512, 1, 16777215)  # 0xFFFFFF

        image = result[0]
        assert image.shape == (1, 512, 512, 3)
        assert np.allclose(image, 1)

    def test_create_batch(self):
        """Create batch of empty images"""
        node = EmptyImage()
        result = node.generate(256, 256, 4, 0)

        image = result[0]
        assert image.shape == (4, 256, 256, 3)


class TestImageFlip:
    """Tests for ImageFlip node"""

    def test_flip_horizontal(self, sample_image_tensor):
        """Flip image horizontally"""
        node = ImageFlip()
        result = node.flip(sample_image_tensor, "horizontal")

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_flip_vertical(self, sample_image_tensor):
        """Flip image vertically"""
        node = ImageFlip()
        result = node.flip(sample_image_tensor, "vertical")

        image = result[0]
        assert image.shape == sample_image_tensor.shape


class TestImageRotate:
    """Tests for ImageRotate node"""

    def test_rotate_90(self, sample_image_tensor):
        """Rotate 90 degrees"""
        node = ImageRotate()
        result = node.rotate(sample_image_tensor, "90")

        image = result[0]
        # Dimensions should swap
        assert image.shape[1] == sample_image_tensor.shape[2]
        assert image.shape[2] == sample_image_tensor.shape[1]

    def test_rotate_180(self, sample_image_tensor):
        """Rotate 180 degrees"""
        node = ImageRotate()
        result = node.rotate(sample_image_tensor, "180")

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_rotate_270(self, sample_image_tensor):
        """Rotate 270 degrees"""
        node = ImageRotate()
        result = node.rotate(sample_image_tensor, "270")

        image = result[0]
        # Dimensions should swap
        assert image.shape[1] == sample_image_tensor.shape[2]
        assert image.shape[2] == sample_image_tensor.shape[1]


class TestImageCrop:
    """Tests for ImageCrop node"""

    def test_crop_basic(self, sample_image_tensor):
        """Basic crop operation"""
        node = ImageCrop()
        result = node.crop(sample_image_tensor, 10, 20, 100, 50)

        image = result[0]
        assert image.shape == (1, 50, 100, 3)

    def test_crop_full(self, sample_image_tensor):
        """Crop full image"""
        node = ImageCrop()
        result = node.crop(sample_image_tensor, 0, 0, 256, 256)

        image = result[0]
        assert image.shape == sample_image_tensor.shape
