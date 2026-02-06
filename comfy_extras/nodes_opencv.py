"""
ComfyRaw - OpenCV Image Processing Nodes
Comprehensive set of image processing nodes using OpenCV
"""

import cv2
import numpy as np
from comfy_cv.image import ImageProcessor
from comfy_cv.types import ensure_uint8, ensure_float32

MAX_RESOLUTION = 16384


# =============================================================================
# Filter Nodes
# =============================================================================

class GaussianBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/filter"

    def apply(self, image, kernel_size, sigma):
        return (ImageProcessor.gaussian_blur(image, kernel_size, sigma),)


class MedianBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/filter"

    def apply(self, image, kernel_size):
        return (ImageProcessor.median_blur(image, kernel_size),)


class BilateralFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "d": ("INT", {"default": 9, "min": 1, "max": 50}),
                "sigma_color": ("FLOAT", {"default": 75.0, "min": 1.0, "max": 500.0}),
                "sigma_space": ("FLOAT", {"default": 75.0, "min": 1.0, "max": 500.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/filter"

    def apply(self, image, d, sigma_color, sigma_space):
        return (ImageProcessor.bilateral_filter(image, d, sigma_color, sigma_space),)


class BoxBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/filter"

    def apply(self, image, kernel_size):
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            blurred = cv2.blur(img, (kernel_size, kernel_size))
            results.append(ensure_float32(blurred))
        return (np.stack(results, axis=0),)


class Sharpen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/filter"

    def apply(self, image, amount):
        return (ImageProcessor.sharpen(image, amount),)


class UnsharpMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "threshold": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/filter"

    def apply(self, image, radius, amount, threshold):
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            blurred = cv2.GaussianBlur(img, (radius, radius), 0)
            diff = cv2.subtract(img, blurred)

            if threshold > 0:
                mask = np.abs(diff) > threshold
                diff = diff * mask

            sharpened = cv2.add(img, (diff * amount).astype(np.uint8))
            results.append(ensure_float32(sharpened))
        return (np.stack(results, axis=0),)


# =============================================================================
# Edge Detection Nodes
# =============================================================================

class CannyEdge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 500.0}),
                "high_threshold": ("FLOAT", {"default": 200.0, "min": 0.0, "max": 500.0}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "detect"
    CATEGORY = "image/edge"

    def detect(self, image, low_threshold, high_threshold):
        return (ImageProcessor.canny_edge(image, low_threshold, high_threshold),)


class SobelEdge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["both", "horizontal", "vertical"],),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 31, "step": 2}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "detect"
    CATEGORY = "image/edge"

    def detect(self, image, direction, kernel_size):
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            if direction == "horizontal":
                sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
            elif direction == "vertical":
                sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
            else:
                sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
                sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
                sobel = np.sqrt(sobel_x**2 + sobel_y**2)

            sobel = np.abs(sobel)
            sobel = (sobel / sobel.max() * 255).astype(np.uint8) if sobel.max() > 0 else sobel.astype(np.uint8)
            results.append(sobel.astype(np.float32) / 255.0)

        return (np.stack(results, axis=0),)


class LaplacianEdge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 31, "step": 2}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "detect"
    CATEGORY = "image/edge"

    def detect(self, image, kernel_size):
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size)
            laplacian = np.abs(laplacian)
            laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8) if laplacian.max() > 0 else laplacian.astype(np.uint8)
            results.append(laplacian.astype(np.float32) / 255.0)

        return (np.stack(results, axis=0),)


class FindContours:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "mode": (["external", "list", "tree"],),
            }
        }

    RETURN_TYPES = ("CONTOURS",)
    FUNCTION = "find"
    CATEGORY = "image/edge"

    def find(self, mask, mode):
        modes = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "tree": cv2.RETR_TREE,
        }

        all_contours = []
        for i in range(mask.shape[0]):
            m = ensure_uint8(mask[i])
            contours, _ = cv2.findContours(m, modes[mode], cv2.CHAIN_APPROX_SIMPLE)
            all_contours.append(contours)

        return (all_contours,)


class DrawContours:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "contours": ("CONTOURS",),
                "color": ("INT", {"default": 0x00FF00, "min": 0, "max": 0xFFFFFF, "display": "color"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "image/edge"

    def draw(self, image, contours, color, thickness):
        r = ((color >> 16) & 0xFF)
        g = ((color >> 8) & 0xFF)
        b = (color & 0xFF)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i]).copy()
            if i < len(contours):
                cv2.drawContours(img, contours[i], -1, (r, g, b), thickness)
            results.append(ensure_float32(img))

        return (np.stack(results, axis=0),)


# =============================================================================
# Morphology Nodes
# =============================================================================

class Erode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/morphology"

    def apply(self, image, kernel_size, iterations):
        return (ImageProcessor.erode(image, kernel_size, iterations),)


class Dilate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/morphology"

    def apply(self, image, kernel_size, iterations):
        return (ImageProcessor.dilate(image, kernel_size, iterations),)


class MorphOpen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/morphology"

    def apply(self, image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            results.append(ensure_float32(opened))
        return (np.stack(results, axis=0),)


class MorphClose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/morphology"

    def apply(self, image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            results.append(ensure_float32(closed))
        return (np.stack(results, axis=0),)


class MorphGradient:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/morphology"

    def apply(self, image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
            results.append(ensure_float32(gradient))
        return (np.stack(results, axis=0),)


# =============================================================================
# Color Processing Nodes
# =============================================================================

class ColorConvert:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "conversion": ([
                    "rgb_to_gray",
                    "rgb_to_hsv",
                    "rgb_to_lab",
                    "hsv_to_rgb",
                    "lab_to_rgb",
                    "gray_to_rgb",
                ],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "image/color"

    def convert(self, image, conversion):
        return (ImageProcessor.color_convert(image, conversion),)


class AdjustBrightness:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "image/color"

    def adjust(self, image, factor):
        return (ImageProcessor.adjust_brightness(image, factor),)


class AdjustContrast:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "image/color"

    def adjust(self, image, factor):
        return (ImageProcessor.adjust_contrast(image, factor),)


class AdjustHueSaturation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue_shift": ("INT", {"default": 0, "min": -180, "max": 180}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "image/color"

    def adjust(self, image, hue_shift, saturation):
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)

            hsv = hsv.astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            results.append(ensure_float32(rgb))

        return (np.stack(results, axis=0),)


class HistogramEqualize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "equalize"
    CATEGORY = "image/color"

    def equalize(self, image):
        return (ImageProcessor.histogram_equalize(image),)


class ColorBalance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "green": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "blue": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "balance"
    CATEGORY = "image/color"

    def balance(self, image, red, green, blue):
        result = image.copy()
        result[..., 0] = np.clip(result[..., 0] * red, 0, 1)
        result[..., 1] = np.clip(result[..., 1] * green, 0, 1)
        result[..., 2] = np.clip(result[..., 2] * blue, 0, 1)
        return (result.astype(np.float32),)


# =============================================================================
# Threshold Nodes
# =============================================================================

class Threshold:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (["binary", "binary_inv", "trunc", "tozero", "tozero_inv", "otsu"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply"
    CATEGORY = "image/threshold"

    def apply(self, image, threshold, method):
        return (ImageProcessor.threshold(image, threshold, 1.0, method),)


class AdaptiveThreshold:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "block_size": ("INT", {"default": 11, "min": 3, "max": 99, "step": 2}),
                "c": ("INT", {"default": 2, "min": -50, "max": 50}),
                "method": (["mean", "gaussian"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply"
    CATEGORY = "image/threshold"

    def apply(self, image, block_size, c, method):
        methods = {
            "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
            "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        }

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            threshed = cv2.adaptiveThreshold(
                img, 255, methods[method],
                cv2.THRESH_BINARY, block_size, c
            )
            results.append(threshed.astype(np.float32) / 255.0)

        return (np.stack(results, axis=0),)


# =============================================================================
# Composite Nodes
# =============================================================================

class Blend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["normal", "add", "multiply", "screen", "overlay"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "image/composite"

    def blend(self, image1, image2, alpha, mode):
        if mode == "normal":
            result = image1 * alpha + image2 * (1 - alpha)
        elif mode == "add":
            result = np.clip(image1 + image2, 0, 1)
        elif mode == "multiply":
            result = image1 * image2
        elif mode == "screen":
            result = 1 - (1 - image1) * (1 - image2)
        elif mode == "overlay":
            mask = image2 < 0.5
            result = np.where(mask, 2 * image1 * image2, 1 - 2 * (1 - image1) * (1 - image2))
        else:
            result = image1 * alpha + image2 * (1 - alpha)

        return (np.clip(result, 0, 1).astype(np.float32),)


class AlphaComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "background": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "image/composite"

    def composite(self, foreground, background, mask):
        # Expand mask to match image channels
        if mask.ndim == 3:
            mask = np.expand_dims(mask, -1)

        result = foreground * mask + background * (1 - mask)
        return (result.astype(np.float32),)


class MaskApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/composite"

    def apply(self, image, mask):
        if mask.ndim == 3:
            mask = np.expand_dims(mask, -1)

        result = image * mask
        return (result.astype(np.float32),)


class ChannelSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("red", "green", "blue")
    FUNCTION = "split"
    CATEGORY = "image/composite"

    def split(self, image):
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        return (r, g, b)


class ChannelMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "red": ("MASK",),
                "green": ("MASK",),
                "blue": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "image/composite"

    def merge(self, red, green, blue):
        result = np.stack([red, green, blue], axis=-1)
        return (result.astype(np.float32),)


# =============================================================================
# Draw Nodes
# =============================================================================

class DrawText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "Hello"}),
                "x": ("INT", {"default": 10, "min": 0, "max": MAX_RESOLUTION}),
                "y": ("INT", {"default": 30, "min": 0, "max": MAX_RESOLUTION}),
                "font_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "color": ("INT", {"default": 0xFFFFFF, "min": 0, "max": 0xFFFFFF, "display": "color"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "image/draw"

    def draw(self, image, text, x, y, font_scale, color, thickness):
        r = ((color >> 16) & 0xFF)
        g = ((color >> 8) & 0xFF)
        b = (color & 0xFF)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i]).copy()
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (r, g, b), thickness)
            results.append(ensure_float32(img))

        return (np.stack(results, axis=0),)


class DrawRectangle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 10, "min": 0, "max": MAX_RESOLUTION}),
                "y": ("INT", {"default": 10, "min": 0, "max": MAX_RESOLUTION}),
                "width": ("INT", {"default": 100, "min": 1, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 100, "min": 1, "max": MAX_RESOLUTION}),
                "color": ("INT", {"default": 0x00FF00, "min": 0, "max": 0xFFFFFF, "display": "color"}),
                "thickness": ("INT", {"default": 2, "min": -1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "image/draw"

    def draw(self, image, x, y, width, height, color, thickness):
        r = ((color >> 16) & 0xFF)
        g = ((color >> 8) & 0xFF)
        b = (color & 0xFF)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i]).copy()
            cv2.rectangle(img, (x, y), (x + width, y + height), (r, g, b), thickness)
            results.append(ensure_float32(img))

        return (np.stack(results, axis=0),)


class DrawCircle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "center_x": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION}),
                "center_y": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION}),
                "radius": ("INT", {"default": 50, "min": 1, "max": MAX_RESOLUTION}),
                "color": ("INT", {"default": 0xFF0000, "min": 0, "max": 0xFFFFFF, "display": "color"}),
                "thickness": ("INT", {"default": 2, "min": -1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "image/draw"

    def draw(self, image, center_x, center_y, radius, color, thickness):
        r = ((color >> 16) & 0xFF)
        g = ((color >> 8) & 0xFF)
        b = (color & 0xFF)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i]).copy()
            cv2.circle(img, (center_x, center_y), radius, (r, g, b), thickness)
            results.append(ensure_float32(img))

        return (np.stack(results, axis=0),)


class DrawLine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "x1": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "y1": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "x2": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION}),
                "y2": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION}),
                "color": ("INT", {"default": 0x0000FF, "min": 0, "max": 0xFFFFFF, "display": "color"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "image/draw"

    def draw(self, image, x1, y1, x2, y2, color, thickness):
        r = ((color >> 16) & 0xFF)
        g = ((color >> 8) & 0xFF)
        b = (color & 0xFF)

        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i]).copy()
            cv2.line(img, (x1, y1), (x2, y2), (r, g, b), thickness)
            results.append(ensure_float32(img))

        return (np.stack(results, axis=0),)


# =============================================================================
# Analysis Nodes
# =============================================================================

class ImageInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("batch", "height", "width", "channels")
    FUNCTION = "info"
    CATEGORY = "image/analysis"

    def info(self, image):
        b, h, w, c = image.shape
        return (b, h, w, c)


class Histogram:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bins": ("INT", {"default": 256, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = "image/analysis"

    def compute(self, image, bins):
        results = []
        for i in range(image.shape[0]):
            img = ensure_uint8(image[i])

            # Create histogram image
            hist_h = 256
            hist_w = 512
            hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

            for c, color in enumerate(colors):
                hist = cv2.calcHist([img], [c], None, [bins], [0, 256])
                hist = hist.flatten()
                if hist.max() > 0:
                    hist = hist / hist.max() * (hist_h - 1)

                for j in range(1, bins):
                    x1 = int((j - 1) * hist_w / bins)
                    x2 = int(j * hist_w / bins)
                    y1 = hist_h - int(hist[j - 1])
                    y2 = hist_h - int(hist[j])
                    cv2.line(hist_img, (x1, y1), (x2, y2), color, 2)

            results.append(ensure_float32(hist_img))

        return (np.stack(results, axis=0),)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    # Filters
    "GaussianBlur": GaussianBlur,
    "MedianBlur": MedianBlur,
    "BilateralFilter": BilateralFilter,
    "BoxBlur": BoxBlur,
    "Sharpen": Sharpen,
    "UnsharpMask": UnsharpMask,

    # Edge Detection
    "CannyEdge": CannyEdge,
    "SobelEdge": SobelEdge,
    "LaplacianEdge": LaplacianEdge,
    "FindContours": FindContours,
    "DrawContours": DrawContours,

    # Morphology
    "Erode": Erode,
    "Dilate": Dilate,
    "MorphOpen": MorphOpen,
    "MorphClose": MorphClose,
    "MorphGradient": MorphGradient,

    # Color
    "ColorConvert": ColorConvert,
    "AdjustBrightness": AdjustBrightness,
    "AdjustContrast": AdjustContrast,
    "AdjustHueSaturation": AdjustHueSaturation,
    "HistogramEqualize": HistogramEqualize,
    "ColorBalance": ColorBalance,

    # Threshold
    "Threshold": Threshold,
    "AdaptiveThreshold": AdaptiveThreshold,

    # Composite
    "Blend": Blend,
    "AlphaComposite": AlphaComposite,
    "MaskApply": MaskApply,
    "ChannelSplit": ChannelSplit,
    "ChannelMerge": ChannelMerge,

    # Draw
    "DrawText": DrawText,
    "DrawRectangle": DrawRectangle,
    "DrawCircle": DrawCircle,
    "DrawLine": DrawLine,

    # Analysis
    "ImageInfo": ImageInfo,
    "Histogram": Histogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Filters
    "GaussianBlur": "Gaussian Blur",
    "MedianBlur": "Median Blur",
    "BilateralFilter": "Bilateral Filter",
    "BoxBlur": "Box Blur",
    "Sharpen": "Sharpen",
    "UnsharpMask": "Unsharp Mask",

    # Edge Detection
    "CannyEdge": "Canny Edge",
    "SobelEdge": "Sobel Edge",
    "LaplacianEdge": "Laplacian Edge",
    "FindContours": "Find Contours",
    "DrawContours": "Draw Contours",

    # Morphology
    "Erode": "Erode",
    "Dilate": "Dilate",
    "MorphOpen": "Morphology Open",
    "MorphClose": "Morphology Close",
    "MorphGradient": "Morphology Gradient",

    # Color
    "ColorConvert": "Color Convert",
    "AdjustBrightness": "Adjust Brightness",
    "AdjustContrast": "Adjust Contrast",
    "AdjustHueSaturation": "Adjust Hue/Saturation",
    "HistogramEqualize": "Histogram Equalize",
    "ColorBalance": "Color Balance",

    # Threshold
    "Threshold": "Threshold",
    "AdaptiveThreshold": "Adaptive Threshold",

    # Composite
    "Blend": "Blend",
    "AlphaComposite": "Alpha Composite",
    "MaskApply": "Mask Apply",
    "ChannelSplit": "Channel Split",
    "ChannelMerge": "Channel Merge",

    # Draw
    "DrawText": "Draw Text",
    "DrawRectangle": "Draw Rectangle",
    "DrawCircle": "Draw Circle",
    "DrawLine": "Draw Line",

    # Analysis
    "ImageInfo": "Image Info",
    "Histogram": "Histogram",
}
