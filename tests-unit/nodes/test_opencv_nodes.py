"""
ComfyRaw - Unit tests for OpenCV nodes in comfy_extras/nodes_opencv.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np

from comfy_extras.nodes_opencv import (
    # Filters
    GaussianBlur,
    MedianBlur,
    BilateralFilter,
    BoxBlur,
    Sharpen,
    UnsharpMask,
    # Edge Detection
    CannyEdge,
    SobelEdge,
    LaplacianEdge,
    # Morphology
    Erode,
    Dilate,
    MorphOpen,
    MorphClose,
    MorphGradient,
    # Color
    ColorConvert,
    AdjustBrightness,
    AdjustContrast,
    AdjustHueSaturation,
    HistogramEqualize,
    ColorBalance,
    # Threshold
    Threshold,
    AdaptiveThreshold,
    # Composite
    Blend,
    AlphaComposite,
    MaskApply,
    ChannelSplit,
    ChannelMerge,
    # Draw
    DrawText,
    DrawRectangle,
    DrawCircle,
    DrawLine,
    # Analysis
    ImageInfo,
    Histogram,
)


@pytest.fixture
def sample_image_tensor():
    """Create a sample IMAGE tensor (B, H, W, C) float32"""
    return np.random.rand(1, 256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_mask_tensor():
    """Create a sample MASK tensor (B, H, W) float32"""
    return np.random.rand(1, 256, 256).astype(np.float32)


class TestFilterNodes:
    """Tests for filter nodes"""

    def test_gaussian_blur(self, sample_image_tensor):
        """Test Gaussian blur node"""
        node = GaussianBlur()
        result = node.apply(sample_image_tensor, 5, 0.0)

        image = result[0]
        assert image.shape == sample_image_tensor.shape
        assert image.dtype == np.float32

    def test_median_blur(self, sample_image_tensor):
        """Test median blur node"""
        node = MedianBlur()
        result = node.apply(sample_image_tensor, 5)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_bilateral_filter(self, sample_image_tensor):
        """Test bilateral filter node"""
        node = BilateralFilter()
        result = node.apply(sample_image_tensor, 9, 75.0, 75.0)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_box_blur(self, sample_image_tensor):
        """Test box blur node"""
        node = BoxBlur()
        result = node.apply(sample_image_tensor, 5)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_sharpen(self, sample_image_tensor):
        """Test sharpen node"""
        node = Sharpen()
        result = node.apply(sample_image_tensor, 1.0)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_unsharp_mask(self, sample_image_tensor):
        """Test unsharp mask node"""
        node = UnsharpMask()
        result = node.apply(sample_image_tensor, 5, 1.0, 0.5)

        image = result[0]
        assert image.shape == sample_image_tensor.shape


class TestEdgeDetectionNodes:
    """Tests for edge detection nodes - return MASK (B, H, W)"""

    def test_canny_edge(self, sample_image_tensor):
        """Test Canny edge detection node"""
        node = CannyEdge()
        result = node.detect(sample_image_tensor, 100.0, 200.0)

        mask = result[0]
        assert mask.ndim == 3  # (B, H, W) mask
        assert mask.dtype == np.float32
        assert mask.shape[:2] == sample_image_tensor.shape[:2]

    def test_sobel_edge(self, sample_image_tensor):
        """Test Sobel edge detection node"""
        node = SobelEdge()
        result = node.detect(sample_image_tensor, "both", 3)

        mask = result[0]
        assert mask.ndim == 3  # MASK output
        assert mask.shape[:2] == sample_image_tensor.shape[:2]

    def test_laplacian_edge(self, sample_image_tensor):
        """Test Laplacian edge detection node"""
        node = LaplacianEdge()
        result = node.detect(sample_image_tensor, 3)

        mask = result[0]
        assert mask.ndim == 3  # MASK output
        assert mask.shape[:2] == sample_image_tensor.shape[:2]


class TestMorphologyNodes:
    """Tests for morphology nodes"""

    def test_erode(self, sample_image_tensor):
        """Test erosion node"""
        node = Erode()
        result = node.apply(sample_image_tensor, 3, 1)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_dilate(self, sample_image_tensor):
        """Test dilation node"""
        node = Dilate()
        result = node.apply(sample_image_tensor, 3, 1)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_morph_open(self, sample_image_tensor):
        """Test morphological opening node"""
        node = MorphOpen()
        result = node.apply(sample_image_tensor, 3)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_morph_close(self, sample_image_tensor):
        """Test morphological closing node"""
        node = MorphClose()
        result = node.apply(sample_image_tensor, 3)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_morph_gradient(self, sample_image_tensor):
        """Test morphological gradient node"""
        node = MorphGradient()
        result = node.apply(sample_image_tensor, 3)

        image = result[0]
        assert image.shape == sample_image_tensor.shape


class TestColorNodes:
    """Tests for color adjustment nodes"""

    def test_color_convert(self, sample_image_tensor):
        """Test color conversion node"""
        node = ColorConvert()
        result = node.convert(sample_image_tensor, "rgb_to_gray")

        image = result[0]
        assert image.shape[3] == 1  # Grayscale

    def test_adjust_brightness(self, sample_image_tensor):
        """Test brightness adjustment node"""
        node = AdjustBrightness()
        result = node.adjust(sample_image_tensor, 1.5)

        image = result[0]
        assert image.shape == sample_image_tensor.shape
        assert np.mean(image) > np.mean(sample_image_tensor)

    def test_adjust_contrast(self, sample_image_tensor):
        """Test contrast adjustment node"""
        node = AdjustContrast()
        result = node.adjust(sample_image_tensor, 2.0)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_adjust_hue_saturation(self, sample_image_tensor):
        """Test hue/saturation adjustment node"""
        node = AdjustHueSaturation()
        result = node.adjust(sample_image_tensor, 0, 1.2)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_histogram_equalize(self, sample_image_tensor):
        """Test histogram equalization node"""
        node = HistogramEqualize()
        result = node.equalize(sample_image_tensor)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_color_balance(self, sample_image_tensor):
        """Test color balance node"""
        node = ColorBalance()
        result = node.balance(sample_image_tensor, 1.0, 1.0, 1.0)

        image = result[0]
        assert image.shape == sample_image_tensor.shape


class TestThresholdNodes:
    """Tests for threshold nodes - return MASK (B, H, W)"""

    def test_threshold(self, sample_image_tensor):
        """Test threshold node"""
        node = Threshold()
        result = node.apply(sample_image_tensor, 0.5, "binary")

        mask = result[0]
        assert mask.ndim == 3  # MASK (B, H, W)
        assert mask.shape[:2] == sample_image_tensor.shape[:2]

    def test_adaptive_threshold(self, sample_image_tensor):
        """Test adaptive threshold node"""
        node = AdaptiveThreshold()
        result = node.apply(sample_image_tensor, 11, 2, "gaussian")

        mask = result[0]
        assert mask.ndim == 3  # MASK


class TestCompositeNodes:
    """Tests for composite nodes"""

    def test_blend(self, sample_image_tensor):
        """Test blend node"""
        node = Blend()
        image2 = np.random.rand(*sample_image_tensor.shape).astype(np.float32)
        result = node.blend(sample_image_tensor, image2, 0.5, "normal")

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_alpha_composite(self, sample_image_tensor):
        """Test alpha composite node"""
        node = AlphaComposite()
        image2 = np.random.rand(*sample_image_tensor.shape).astype(np.float32)
        mask = np.random.rand(1, 256, 256).astype(np.float32)
        result = node.composite(sample_image_tensor, image2, mask)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_mask_apply(self, sample_image_tensor, sample_mask_tensor):
        """Test mask apply node"""
        node = MaskApply()
        result = node.apply(sample_image_tensor, sample_mask_tensor)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_channel_split(self, sample_image_tensor):
        """Test channel split node"""
        node = ChannelSplit()
        result = node.split(sample_image_tensor)

        r, g, b = result
        assert r.shape == (1, 256, 256)
        assert g.shape == (1, 256, 256)
        assert b.shape == (1, 256, 256)

    def test_channel_merge(self, sample_mask_tensor):
        """Test channel merge node"""
        node = ChannelMerge()
        result = node.merge(sample_mask_tensor, sample_mask_tensor, sample_mask_tensor)

        image = result[0]
        assert image.shape[3] == 3


class TestDrawNodes:
    """Tests for drawing nodes"""

    def test_draw_text(self, sample_image_tensor):
        """Test draw text node"""
        node = DrawText()
        result = node.draw(sample_image_tensor, "Hello", 10, 50, 1.0, 16777215, 2)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_draw_rectangle(self, sample_image_tensor):
        """Test draw rectangle node"""
        node = DrawRectangle()
        result = node.draw(sample_image_tensor, 10, 10, 100, 100, 16777215, 2)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_draw_circle(self, sample_image_tensor):
        """Test draw circle node"""
        node = DrawCircle()
        result = node.draw(sample_image_tensor, 128, 128, 50, 16777215, 2)

        image = result[0]
        assert image.shape == sample_image_tensor.shape

    def test_draw_line(self, sample_image_tensor):
        """Test draw line node"""
        node = DrawLine()
        result = node.draw(sample_image_tensor, 0, 0, 255, 255, 16777215, 2)

        image = result[0]
        assert image.shape == sample_image_tensor.shape


class TestAnalysisNodes:
    """Tests for analysis nodes"""

    def test_image_info(self, sample_image_tensor):
        """Test image info node"""
        node = ImageInfo()
        result = node.info(sample_image_tensor)

        # Returns: batch, height, width, channels
        assert len(result) == 4
        batch, height, width, channels = result

        assert batch == 1
        assert height == 256
        assert width == 256
        assert channels == 3

    def test_histogram(self, sample_image_tensor):
        """Test histogram node"""
        node = Histogram()
        result = node.compute(sample_image_tensor, 256)

        histogram = result[0]
        # Histogram should be an image visualization
        assert histogram.ndim == 4
