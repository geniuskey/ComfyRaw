"""
ComfyRaw - Utility functions for image processing
"""

import numpy as np
from PIL import Image
import logging
import os


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (H, W, C) float32 RGB 0-1"""
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.ndim == 2:  # grayscale
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:  # RGBA
        arr = arr[:, :, :3]
    return arr


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB format"""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        return image[:, :, :3]
    return image


def ensure_batch(image: np.ndarray) -> np.ndarray:
    """Ensure image has batch dimension (B, H, W, C)"""
    if image.ndim == 3:
        return np.expand_dims(image, 0)
    return image


def unbatch(images: np.ndarray) -> list:
    """Convert batch to list of individual images"""
    if images.ndim == 4:
        return [images[i] for i in range(images.shape[0])]
    return [images]


def get_image_size(image: np.ndarray) -> tuple:
    """Get image size as (width, height)"""
    if image.ndim == 4:
        return (image.shape[2], image.shape[1])
    return (image.shape[1], image.shape[0])


def resize_image(image: np.ndarray, width: int, height: int, method: str = "bilinear") -> np.ndarray:
    """Resize image using PIL"""
    import cv2

    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }

    interp = interpolation_methods.get(method, cv2.INTER_LINEAR)

    was_batch = image.ndim == 4
    if was_batch:
        results = []
        for i in range(image.shape[0]):
            img = image[i]
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
                resized = cv2.resize(img, (width, height), interpolation=interp)
                resized = resized.astype(np.float32) / 255.0
            else:
                resized = cv2.resize(img, (width, height), interpolation=interp)
            results.append(resized)
        return np.stack(results, axis=0)
    else:
        if image.dtype == np.float32:
            img = (image * 255).astype(np.uint8)
            resized = cv2.resize(img, (width, height), interpolation=interp)
            return resized.astype(np.float32) / 255.0
        return cv2.resize(image, (width, height), interpolation=interp)


class ProgressBar:
    """Simple progress bar for tracking operations"""

    def __init__(self, total: int):
        self.total = total
        self.current = 0

    def update(self, value: int = 1):
        self.current += value
        if self.total > 0:
            percent = (self.current / self.total) * 100
            logging.debug(f"Progress: {percent:.1f}%")

    def update_absolute(self, value: int, total: int = None):
        if total is not None:
            self.total = total
        self.current = value


def common_upscale(samples, width, height, upscale_method, crop):
    """Upscale images with optional cropping"""
    import cv2

    if crop == "center":
        old_width = samples.shape[2] if samples.ndim == 4 else samples.shape[1]
        old_height = samples.shape[1] if samples.ndim == 4 else samples.shape[0]

        old_aspect = old_width / old_height
        new_aspect = width / height

        if old_aspect > new_aspect:
            # Crop width
            new_width = int(old_height * new_aspect)
            x_start = (old_width - new_width) // 2
            if samples.ndim == 4:
                samples = samples[:, :, x_start:x_start+new_width, :]
            else:
                samples = samples[:, x_start:x_start+new_width, :]
        else:
            # Crop height
            new_height = int(old_width / new_aspect)
            y_start = (old_height - new_height) // 2
            if samples.ndim == 4:
                samples = samples[:, y_start:y_start+new_height, :, :]
            else:
                samples = samples[y_start:y_start+new_height, :, :]

    return resize_image(samples, width, height, upscale_method)
