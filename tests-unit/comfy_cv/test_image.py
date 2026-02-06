"""
ComfyRaw - Unit tests for comfy_cv.image module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from comfy_cv.image import ImageProcessor


class TestImageLoad:
    """Tests for ImageProcessor.load"""

    def test_load_png(self, temp_image_path):
        """Load a PNG image"""
        result = ImageProcessor.load(temp_image_path)

        assert result.ndim == 4
        assert result.dtype == np.float32
        assert result.shape[0] == 1  # batch size
        assert result.shape[3] == 3  # RGB channels

    def test_load_nonexistent(self):
        """Loading non-existent file should raise ValueError"""
        with pytest.raises(ValueError):
            ImageProcessor.load("/nonexistent/path/image.png")


class TestImageSave:
    """Tests for ImageProcessor.save"""

    def test_save_png(self, sample_image, temp_dir):
        """Save image as PNG"""
        path = os.path.join(temp_dir, "test.png")
        ImageProcessor.save(sample_image, path)

        assert os.path.exists(path)

        # Verify we can load it back
        loaded = ImageProcessor.load(path)
        assert loaded.shape == sample_image.shape

    def test_save_jpg(self, sample_image, temp_dir):
        """Save image as JPEG"""
        path = os.path.join(temp_dir, "test.jpg")
        ImageProcessor.save(sample_image, path, quality=90)

        assert os.path.exists(path)

    def test_save_batch_first_frame(self, sample_batch, temp_dir):
        """Save batch should save first frame"""
        path = os.path.join(temp_dir, "test.png")
        ImageProcessor.save(sample_batch, path)

        assert os.path.exists(path)


class TestImageResize:
    """Tests for ImageProcessor.resize"""

    def test_resize_basic(self, sample_image):
        """Basic resize to specific dimensions"""
        result = ImageProcessor.resize(sample_image, 128, 64)

        assert result.shape == (1, 64, 128, 3)
        assert result.dtype == np.float32

    def test_resize_batch(self, sample_batch):
        """Resize batch of images"""
        result = ImageProcessor.resize(sample_batch, 64, 64)

        assert result.shape == (4, 64, 64, 3)

    def test_resize_methods(self, sample_image):
        """Test different interpolation methods"""
        for method in ["nearest", "bilinear", "bicubic", "lanczos", "area"]:
            result = ImageProcessor.resize(sample_image, 128, 128, method=method)
            assert result.shape == (1, 128, 128, 3)


class TestImageCrop:
    """Tests for ImageProcessor.crop"""

    def test_crop_basic(self, sample_image):
        """Basic crop operation"""
        result = ImageProcessor.crop(sample_image, x=10, y=20, width=100, height=50)

        assert result.shape == (1, 50, 100, 3)

    def test_crop_batch(self, sample_batch):
        """Crop batch of images"""
        result = ImageProcessor.crop(sample_batch, x=0, y=0, width=64, height=64)

        assert result.shape == (4, 64, 64, 3)


class TestImageRotate:
    """Tests for ImageProcessor.rotate"""

    def test_rotate_90(self, sample_image):
        """Rotate 90 degrees"""
        result = ImageProcessor.rotate(sample_image, 90.0, expand=True)

        # With expand, dimensions should swap
        assert result.shape[1] == sample_image.shape[2]
        assert result.shape[2] == sample_image.shape[1]

    def test_rotate_no_expand(self, sample_image):
        """Rotate without expanding canvas"""
        result = ImageProcessor.rotate(sample_image, 45.0, expand=False)

        assert result.shape == sample_image.shape


class TestImageFlip:
    """Tests for ImageProcessor.flip"""

    def test_flip_horizontal(self, gradient_image):
        """Flip horizontally"""
        result = ImageProcessor.flip(gradient_image, horizontal=True, vertical=False)

        # First column should now be brightest
        assert result[0, 0, 0, 0] > result[0, 0, -1, 0]

    def test_flip_vertical(self, sample_image):
        """Flip vertically"""
        result = ImageProcessor.flip(sample_image, horizontal=False, vertical=True)

        assert result.shape == sample_image.shape

    def test_flip_both(self, sample_image):
        """Flip both directions"""
        result = ImageProcessor.flip(sample_image, horizontal=True, vertical=True)

        assert result.shape == sample_image.shape


class TestGaussianBlur:
    """Tests for ImageProcessor.gaussian_blur"""

    def test_blur_basic(self, checkerboard_image):
        """Basic Gaussian blur"""
        result = ImageProcessor.gaussian_blur(checkerboard_image, kernel_size=5)

        assert result.shape == checkerboard_image.shape
        # Blurred image should have less variance
        assert np.std(result) < np.std(checkerboard_image)

    def test_blur_even_kernel(self, sample_image):
        """Even kernel size should be converted to odd"""
        result = ImageProcessor.gaussian_blur(sample_image, kernel_size=4)
        assert result.shape == sample_image.shape


class TestMedianBlur:
    """Tests for ImageProcessor.median_blur"""

    def test_median_blur(self, sample_image):
        """Basic median blur"""
        result = ImageProcessor.median_blur(sample_image, kernel_size=5)
        assert result.shape == sample_image.shape


class TestBilateralFilter:
    """Tests for ImageProcessor.bilateral_filter"""

    def test_bilateral(self, sample_image):
        """Basic bilateral filter"""
        result = ImageProcessor.bilateral_filter(sample_image, d=9, sigma_color=75, sigma_space=75)
        assert result.shape == sample_image.shape


class TestCannyEdge:
    """Tests for ImageProcessor.canny_edge"""

    def test_canny_basic(self, checkerboard_image):
        """Canny edge detection"""
        result = ImageProcessor.canny_edge(checkerboard_image, 100, 200)

        # Output should be a mask (no color channel or single channel)
        assert result.shape[0] == checkerboard_image.shape[0]
        assert result.shape[1] == checkerboard_image.shape[1]
        assert result.shape[2] == checkerboard_image.shape[2]
        # Should contain edges (non-zero values)
        assert np.max(result) > 0


class TestThreshold:
    """Tests for ImageProcessor.threshold"""

    def test_binary_threshold(self, gradient_image):
        """Binary threshold"""
        result = ImageProcessor.threshold(gradient_image, thresh=0.5)

        # Should only have 0 and 1 values
        unique_values = np.unique(result)
        assert len(unique_values) <= 2

    def test_threshold_methods(self, gradient_image):
        """Test different threshold methods"""
        for method in ["binary", "binary_inv", "trunc", "tozero"]:
            result = ImageProcessor.threshold(gradient_image, thresh=0.5, method=method)
            assert result.shape[:3] == gradient_image.shape[:3]


class TestMorphology:
    """Tests for morphological operations"""

    def test_erode(self, sample_image):
        """Erosion operation"""
        result = ImageProcessor.erode(sample_image, kernel_size=3)
        assert result.shape == sample_image.shape

    def test_dilate(self, sample_image):
        """Dilation operation"""
        result = ImageProcessor.dilate(sample_image, kernel_size=3)
        assert result.shape == sample_image.shape


class TestColorConvert:
    """Tests for ImageProcessor.color_convert"""

    def test_rgb_to_gray(self, sample_image):
        """Convert RGB to grayscale"""
        result = ImageProcessor.color_convert(sample_image, "rgb_to_gray")

        assert result.shape[3] == 1  # Single channel

    def test_rgb_to_hsv(self, sample_image):
        """Convert RGB to HSV"""
        result = ImageProcessor.color_convert(sample_image, "rgb_to_hsv")

        assert result.shape == sample_image.shape

    def test_invalid_conversion(self, sample_image):
        """Invalid conversion should raise ValueError"""
        with pytest.raises(ValueError):
            ImageProcessor.color_convert(sample_image, "invalid_conversion")


class TestAdjustments:
    """Tests for brightness/contrast adjustments"""

    def test_adjust_brightness(self, sample_image):
        """Brightness adjustment"""
        brighter = ImageProcessor.adjust_brightness(sample_image, 1.5)
        darker = ImageProcessor.adjust_brightness(sample_image, 0.5)

        assert np.mean(brighter) > np.mean(sample_image)
        assert np.mean(darker) < np.mean(sample_image)

    def test_adjust_contrast(self, sample_image):
        """Contrast adjustment"""
        high_contrast = ImageProcessor.adjust_contrast(sample_image, 2.0)
        low_contrast = ImageProcessor.adjust_contrast(sample_image, 0.5)

        assert np.std(high_contrast) > np.std(low_contrast)


class TestBlend:
    """Tests for ImageProcessor.blend"""

    def test_blend_50_50(self, black_image, white_image):
        """50/50 blend should produce gray"""
        result = ImageProcessor.blend(black_image, white_image, 0.5)

        assert abs(np.mean(result) - 0.5) < 0.01

    def test_blend_100_0(self, black_image, white_image):
        """100/0 blend should produce first image"""
        result = ImageProcessor.blend(black_image, white_image, 1.0)

        np.testing.assert_array_almost_equal(result, black_image)


class TestInvert:
    """Tests for ImageProcessor.invert"""

    def test_invert_black_to_white(self, black_image):
        """Inverting black should produce white"""
        result = ImageProcessor.invert(black_image)

        np.testing.assert_array_almost_equal(result, np.ones_like(black_image))

    def test_invert_white_to_black(self, white_image):
        """Inverting white should produce black"""
        result = ImageProcessor.invert(white_image)

        np.testing.assert_array_almost_equal(result, np.zeros_like(white_image))


class TestSharpen:
    """Tests for ImageProcessor.sharpen"""

    def test_sharpen_basic(self, sample_image):
        """Basic sharpening"""
        result = ImageProcessor.sharpen(sample_image, amount=1.0)

        assert result.shape == sample_image.shape


class TestHistogramEqualize:
    """Tests for ImageProcessor.histogram_equalize"""

    def test_equalize(self, sample_image):
        """Histogram equalization"""
        result = ImageProcessor.histogram_equalize(sample_image)

        assert result.shape == sample_image.shape
