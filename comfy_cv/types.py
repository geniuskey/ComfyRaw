"""
ComfyRaw - Type definitions for OpenCV processing
"""

import numpy as np
from typing import List, Union, Tuple, TypeAlias

# Image type: (B, H, W, C) float32 RGB normalized 0-1
CVImage: TypeAlias = np.ndarray

# Mask type: (B, H, W) float32 normalized 0-1
CVMask: TypeAlias = np.ndarray

# Video type: List of CVImage frames
CVVideo: TypeAlias = List[np.ndarray]

# Contours type
CVContours: TypeAlias = List[np.ndarray]

# Keypoints type
CVKeypoints: TypeAlias = List[Tuple[float, float]]

# Transformation matrix
CVMatrix: TypeAlias = np.ndarray


def ensure_float32(image: np.ndarray) -> np.ndarray:
    """Ensure image is float32 normalized 0-1"""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    elif image.dtype == np.float64:
        return image.astype(np.float32)
    return image


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 0-255"""
    if image.dtype == np.float32 or image.dtype == np.float64:
        return (image * 255).clip(0, 255).astype(np.uint8)
    return image


def ensure_batch(image: np.ndarray) -> np.ndarray:
    """Ensure image has batch dimension (B, H, W, C)"""
    if image.ndim == 2:
        # Grayscale without batch or channel: (H, W) -> (1, H, W, 1)
        return np.expand_dims(np.expand_dims(image, -1), 0)
    elif image.ndim == 3:
        return np.expand_dims(image, 0)
    return image


def unbatch(images: np.ndarray) -> List[np.ndarray]:
    """Convert batch to list of individual images"""
    if images.ndim == 4:
        return [images[i] for i in range(images.shape[0])]
    return [images]


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR for OpenCV"""
    if image.ndim == 4:
        return image[..., ::-1]
    return image[..., ::-1]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB from OpenCV"""
    if image.ndim == 4:
        return image[..., ::-1]
    return image[..., ::-1]


def validate_image(image: np.ndarray) -> None:
    """Validate image tensor format (B, H, W, C) float32"""
    if image.ndim != 4:
        raise ValueError(f"Image must be 4D (B, H, W, C), got {image.ndim}D")
    if image.dtype != np.float32:
        raise ValueError(f"Image must be float32, got {image.dtype}")
    if image.shape[3] not in (1, 3, 4):
        raise ValueError(f"Image must have 1, 3, or 4 channels, got {image.shape[3]}")


def validate_mask(mask: np.ndarray) -> None:
    """Validate mask tensor format (B, H, W) float32"""
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 3D (B, H, W), got {mask.ndim}D")
    if mask.dtype != np.float32:
        raise ValueError(f"Mask must be float32, got {mask.dtype}")
