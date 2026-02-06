"""
ComfyRaw - Test fixtures for comfy_cv module
"""

import pytest
import numpy as np
import os
import tempfile
from PIL import Image


@pytest.fixture
def sample_image():
    """Create a sample RGB image (1, 256, 256, 3) float32"""
    return np.random.rand(1, 256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_batch():
    """Create a batch of RGB images (4, 128, 128, 3) float32"""
    return np.random.rand(4, 128, 128, 3).astype(np.float32)


@pytest.fixture
def sample_mask():
    """Create a sample mask (1, 256, 256) float32"""
    return np.random.rand(1, 256, 256).astype(np.float32)


@pytest.fixture
def grayscale_image():
    """Create a grayscale image (1, 256, 256, 1) float32"""
    return np.random.rand(1, 256, 256, 1).astype(np.float32)


@pytest.fixture
def rgba_image():
    """Create an RGBA image (1, 256, 256, 4) float32"""
    return np.random.rand(1, 256, 256, 4).astype(np.float32)


@pytest.fixture
def black_image():
    """Create a black image (1, 256, 256, 3) float32"""
    return np.zeros((1, 256, 256, 3), dtype=np.float32)


@pytest.fixture
def white_image():
    """Create a white image (1, 256, 256, 3) float32"""
    return np.ones((1, 256, 256, 3), dtype=np.float32)


@pytest.fixture
def gradient_image():
    """Create a horizontal gradient image (1, 256, 256, 3) float32"""
    gradient = np.linspace(0, 1, 256).reshape(1, 1, 256, 1)
    gradient = np.broadcast_to(gradient, (1, 256, 256, 3)).copy()
    return gradient.astype(np.float32)


@pytest.fixture
def checkerboard_image():
    """Create a checkerboard pattern (1, 256, 256, 3) float32"""
    img = np.zeros((1, 256, 256, 3), dtype=np.float32)
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i // 32 + j // 32) % 2 == 0:
                img[0, i:i+32, j:j+32, :] = 1.0
    return img


@pytest.fixture
def temp_image_path(sample_image):
    """Create a temporary image file and return its path"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name

    # Convert to PIL and save
    img_uint8 = (sample_image[0] * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pil_img.save(path)

    yield path

    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
