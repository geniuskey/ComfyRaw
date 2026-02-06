"""
ComfyRaw - Image processing utilities using OpenCV
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
from .types import CVImage, CVMask, ensure_float32, ensure_uint8, ensure_batch


class ImageProcessor:
    """Static methods for image processing operations"""

    # Interpolation methods mapping
    INTERPOLATION = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }

    # Border types mapping
    BORDER_TYPES = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
        "wrap": cv2.BORDER_WRAP,
    }

    @staticmethod
    def load(path: str) -> CVImage:
        """Load image from file path"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Convert BGR to RGB
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to float32
        img = img.astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(img, 0)

    @staticmethod
    def save(image: CVImage, path: str, quality: int = 95) -> None:
        """Save image to file path"""
        if image.ndim == 4:
            image = image[0]

        # Convert to uint8
        img = ensure_uint8(image)

        # Convert RGB to BGR
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Determine save parameters based on extension
        ext = path.lower().split('.')[-1]
        params = []
        if ext in ['jpg', 'jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == 'png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - quality // 11]
        elif ext == 'webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]

        cv2.imwrite(path, img, params)

    @staticmethod
    def resize(image: CVImage, width: int, height: int,
               method: str = "bilinear") -> CVImage:
        """Resize image to specified dimensions"""
        interp = ImageProcessor.INTERPOLATION.get(method, cv2.INTER_LINEAR)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            resized = cv2.resize(img, (width, height), interpolation=interp)
            results.append(ensure_float32(resized))

        return np.stack(results, axis=0)

    @staticmethod
    def crop(image: CVImage, x: int, y: int, width: int, height: int) -> CVImage:
        """Crop image to specified region"""
        return image[:, y:y+height, x:x+width, :]

    @staticmethod
    def rotate(image: CVImage, angle: float,
               expand: bool = False, fill_color: Tuple = (0, 0, 0)) -> CVImage:
        """Rotate image by specified angle in degrees"""
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            h, w = img.shape[:2]
            center = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            if expand:
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int(h * sin + w * cos)
                new_h = int(h * cos + w * sin)
                M[0, 2] += (new_w - w) / 2
                M[1, 2] += (new_h - h) / 2
                w, h = new_w, new_h

            rotated = cv2.warpAffine(img, M, (w, h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=fill_color)
            results.append(ensure_float32(rotated))

        return np.stack(results, axis=0)

    @staticmethod
    def flip(image: CVImage, horizontal: bool = True, vertical: bool = False) -> CVImage:
        """Flip image horizontally and/or vertically"""
        results = []
        for i in range(image.shape[0]):
            img = image[i].copy()
            if horizontal and vertical:
                img = cv2.flip(img, -1)
            elif horizontal:
                img = cv2.flip(img, 1)
            elif vertical:
                img = cv2.flip(img, 0)
            results.append(img)

        return np.stack(results, axis=0)

    @staticmethod
    def gaussian_blur(image: CVImage, kernel_size: int, sigma: float = 0) -> CVImage:
        """Apply Gaussian blur"""
        if kernel_size % 2 == 0:
            kernel_size += 1

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            results.append(ensure_float32(blurred))

        return np.stack(results, axis=0)

    @staticmethod
    def median_blur(image: CVImage, kernel_size: int) -> CVImage:
        """Apply median blur"""
        if kernel_size % 2 == 0:
            kernel_size += 1

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            blurred = cv2.medianBlur(img, kernel_size)
            results.append(ensure_float32(blurred))

        return np.stack(results, axis=0)

    @staticmethod
    def bilateral_filter(image: CVImage, d: int, sigma_color: float,
                         sigma_space: float) -> CVImage:
        """Apply bilateral filter"""
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
            results.append(ensure_float32(filtered))

        return np.stack(results, axis=0)

    @staticmethod
    def canny_edge(image: CVImage, low_threshold: float,
                   high_threshold: float) -> CVMask:
        """Detect edges using Canny algorithm"""
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img, low_threshold, high_threshold)
            results.append(edges.astype(np.float32) / 255.0)

        return np.stack(results, axis=0)

    @staticmethod
    def threshold(image: CVImage, thresh: float, max_val: float = 1.0,
                  method: str = "binary") -> CVMask:
        """Apply threshold to image"""
        methods = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV,
            "otsu": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        }

        thresh_type = methods.get(method, cv2.THRESH_BINARY)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            _, threshed = cv2.threshold(img, int(thresh * 255),
                                        int(max_val * 255), thresh_type)
            results.append(threshed.astype(np.float32) / 255.0)

        return np.stack(results, axis=0)

    @staticmethod
    def erode(image: CVImage, kernel_size: int, iterations: int = 1) -> CVImage:
        """Apply morphological erosion"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            eroded = cv2.erode(img, kernel, iterations=iterations)
            results.append(ensure_float32(eroded))

        return np.stack(results, axis=0)

    @staticmethod
    def dilate(image: CVImage, kernel_size: int, iterations: int = 1) -> CVImage:
        """Apply morphological dilation"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            dilated = cv2.dilate(img, kernel, iterations=iterations)
            results.append(ensure_float32(dilated))

        return np.stack(results, axis=0)

    @staticmethod
    def color_convert(image: CVImage, conversion: str) -> CVImage:
        """Convert color space"""
        conversions = {
            "rgb_to_gray": cv2.COLOR_RGB2GRAY,
            "rgb_to_hsv": cv2.COLOR_RGB2HSV,
            "rgb_to_lab": cv2.COLOR_RGB2LAB,
            "hsv_to_rgb": cv2.COLOR_HSV2RGB,
            "lab_to_rgb": cv2.COLOR_LAB2RGB,
            "gray_to_rgb": cv2.COLOR_GRAY2RGB,
        }

        code = conversions.get(conversion.lower())
        if code is None:
            raise ValueError(f"Unknown color conversion: {conversion}")

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            converted = cv2.cvtColor(img, code)
            if converted.ndim == 2:
                converted = np.expand_dims(converted, -1)
            results.append(ensure_float32(converted))

        return np.stack(results, axis=0)

    @staticmethod
    def adjust_brightness(image: CVImage, factor: float) -> CVImage:
        """Adjust image brightness"""
        return np.clip(image * factor, 0, 1).astype(np.float32)

    @staticmethod
    def adjust_contrast(image: CVImage, factor: float) -> CVImage:
        """Adjust image contrast"""
        mean = np.mean(image, axis=(1, 2, 3), keepdims=True)
        return np.clip((image - mean) * factor + mean, 0, 1).astype(np.float32)

    @staticmethod
    def blend(image1: CVImage, image2: CVImage, alpha: float) -> CVImage:
        """Blend two images"""
        return np.clip(image1 * alpha + image2 * (1 - alpha), 0, 1).astype(np.float32)

    @staticmethod
    def invert(image: CVImage) -> CVImage:
        """Invert image colors"""
        return (1.0 - image).astype(np.float32)

    @staticmethod
    def sharpen(image: CVImage, amount: float = 1.0) -> CVImage:
        """Sharpen image using unsharp mask"""
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            blurred = cv2.GaussianBlur(img, (0, 0), 3)
            sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
            results.append(ensure_float32(sharpened))

        return np.stack(results, axis=0)

    @staticmethod
    def histogram_equalize(image: CVImage) -> CVImage:
        """Apply histogram equalization"""
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            if img.ndim == 3:
                # Convert to LAB and equalize L channel
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                equalized = cv2.equalizeHist(img)
            results.append(ensure_float32(equalized))

        return np.stack(results, axis=0)
