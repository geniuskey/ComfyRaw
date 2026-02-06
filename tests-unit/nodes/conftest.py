"""
ComfyRaw - Test fixtures for nodes tests
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
import tempfile
from PIL import Image


@pytest.fixture
def sample_image_tensor():
    """Create a sample IMAGE tensor (B, H, W, C) float32"""
    return np.random.rand(1, 256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_mask_tensor():
    """Create a sample MASK tensor (B, H, W) float32"""
    return np.random.rand(1, 256, 256).astype(np.float32)


@pytest.fixture
def sample_batch_tensor():
    """Create a batch IMAGE tensor (4, 128, 128, 3) float32"""
    return np.random.rand(4, 128, 128, 3).astype(np.float32)


@pytest.fixture
def input_dir(tmp_path):
    """Create temporary input directory with test images"""
    input_path = tmp_path / "input"
    input_path.mkdir()

    # Create a test image
    img = Image.new("RGB", (256, 256), color=(128, 64, 32))
    img.save(input_path / "test_image.png")

    # Create a test image with alpha
    img_rgba = Image.new("RGBA", (256, 256), color=(128, 64, 32, 200))
    img_rgba.save(input_path / "test_image_alpha.png")

    return str(input_path)


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory"""
    output_path = tmp_path / "output"
    output_path.mkdir()
    return str(output_path)
