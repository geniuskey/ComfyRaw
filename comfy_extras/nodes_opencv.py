"""
ComfyRaw - OpenCV Image Processing Nodes
All nodes operate on RAW_IMAGE type: {"image": np.ndarray (H,W) float32 0-1, "bit_depth": int}
"""

import cv2
import numpy as np

MAX_RESOLUTION = 16384


def _to_uint8(gray):
    return (gray * 255).clip(0, 255).astype(np.uint8)


def _to_float32(img):
    return img.astype(np.float32) / 255.0


def _wrap(image, bit_depth):
    return {"image": image.clip(0.0, 1.0).astype(np.float32), "bit_depth": bit_depth}


# =============================================================================
# Filter Nodes
# =============================================================================

class GaussianBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/filter"

    def apply(self, raw_image, kernel_size, sigma):
        gray = raw_image["image"]
        if kernel_size % 2 == 0:
            kernel_size += 1
        img = _to_uint8(gray)
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        return (_wrap(_to_float32(blurred), raw_image["bit_depth"]),)


class MedianBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/filter"

    def apply(self, raw_image, kernel_size):
        gray = raw_image["image"]
        if kernel_size % 2 == 0:
            kernel_size += 1
        img = _to_uint8(gray)
        blurred = cv2.medianBlur(img, kernel_size)
        return (_wrap(_to_float32(blurred), raw_image["bit_depth"]),)


class BilateralFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "d": ("INT", {"default": 9, "min": 1, "max": 50}),
                "sigma_color": ("FLOAT", {"default": 75.0, "min": 1.0, "max": 500.0}),
                "sigma_space": ("FLOAT", {"default": 75.0, "min": 1.0, "max": 500.0}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/filter"

    def apply(self, raw_image, d, sigma_color, sigma_space):
        gray = raw_image["image"]
        img = _to_uint8(gray)
        filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        return (_wrap(_to_float32(filtered), raw_image["bit_depth"]),)


class BoxBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/filter"

    def apply(self, raw_image, kernel_size):
        gray = raw_image["image"]
        img = _to_uint8(gray)
        blurred = cv2.blur(img, (kernel_size, kernel_size))
        return (_wrap(_to_float32(blurred), raw_image["bit_depth"]),)


class Sharpen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/filter"

    def apply(self, raw_image, amount):
        gray = raw_image["image"]
        img = _to_uint8(gray)
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        return (_wrap(_to_float32(sharpened), raw_image["bit_depth"]),)


class UnsharpMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "radius": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "threshold": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/filter"

    def apply(self, raw_image, radius, amount, threshold):
        gray = raw_image["image"]
        img = _to_uint8(gray)
        if radius % 2 == 0:
            radius += 1
        blurred = cv2.GaussianBlur(img, (radius, radius), 0)
        diff = cv2.subtract(img, blurred)
        if threshold > 0:
            mask = np.abs(diff.astype(np.int16)) > threshold
            diff = (diff * mask).astype(np.uint8)
        sharpened = cv2.add(img, (diff * amount).astype(np.uint8))
        return (_wrap(_to_float32(sharpened), raw_image["bit_depth"]),)


# =============================================================================
# Edge Detection Nodes
# =============================================================================

class CannyEdge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "low_threshold": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 500.0}),
                "high_threshold": ("FLOAT", {"default": 200.0, "min": 0.0, "max": 500.0}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "detect"
    CATEGORY = "opencv/edge"

    def detect(self, raw_image, low_threshold, high_threshold):
        gray = raw_image["image"]
        img = _to_uint8(gray)
        edges = cv2.Canny(img, low_threshold, high_threshold)
        return (_wrap(_to_float32(edges), raw_image["bit_depth"]),)


class SobelEdge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "direction": (["both", "horizontal", "vertical"],),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 31, "step": 2}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "detect"
    CATEGORY = "opencv/edge"

    def detect(self, raw_image, direction, kernel_size):
        gray = raw_image["image"]
        img = _to_uint8(gray)

        if direction == "horizontal":
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
        elif direction == "vertical":
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
        else:
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)

        sobel = np.abs(sobel)
        if sobel.max() > 0:
            sobel = sobel / sobel.max()
        return (_wrap(sobel.astype(np.float32), raw_image["bit_depth"]),)


class LaplacianEdge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 31, "step": 2}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "detect"
    CATEGORY = "opencv/edge"

    def detect(self, raw_image, kernel_size):
        gray = raw_image["image"]
        img = _to_uint8(gray)
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size)
        laplacian = np.abs(laplacian)
        if laplacian.max() > 0:
            laplacian = laplacian / laplacian.max()
        return (_wrap(laplacian.astype(np.float32), raw_image["bit_depth"]),)


class FindContours:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "mode": (["external", "list", "tree"],),
            }
        }

    RETURN_TYPES = ("CONTOURS",)
    FUNCTION = "find"
    CATEGORY = "opencv/edge"

    def find(self, raw_image, mode):
        modes = {
            "external": cv2.RETR_EXTERNAL,
            "list": cv2.RETR_LIST,
            "tree": cv2.RETR_TREE,
        }
        gray = raw_image["image"]
        img = _to_uint8(gray)
        contours, _ = cv2.findContours(img, modes[mode], cv2.CHAIN_APPROX_SIMPLE)
        return (contours,)


class DrawContours:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "contours": ("CONTOURS",),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "opencv/edge"

    def draw(self, raw_image, contours, intensity, thickness):
        gray = raw_image["image"]
        img = _to_uint8(gray).copy()
        val = int(intensity * 255)
        cv2.drawContours(img, contours, -1, val, thickness)
        return (_wrap(_to_float32(img), raw_image["bit_depth"]),)


# =============================================================================
# Morphology Nodes
# =============================================================================

class Erode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/morphology"

    def apply(self, raw_image, kernel_size, iterations):
        gray = raw_image["image"]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = _to_uint8(gray)
        eroded = cv2.erode(img, kernel, iterations=iterations)
        return (_wrap(_to_float32(eroded), raw_image["bit_depth"]),)


class Dilate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/morphology"

    def apply(self, raw_image, kernel_size, iterations):
        gray = raw_image["image"]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = _to_uint8(gray)
        dilated = cv2.dilate(img, kernel, iterations=iterations)
        return (_wrap(_to_float32(dilated), raw_image["bit_depth"]),)


class MorphOpen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/morphology"

    def apply(self, raw_image, kernel_size):
        gray = raw_image["image"]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = _to_uint8(gray)
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return (_wrap(_to_float32(opened), raw_image["bit_depth"]),)


class MorphClose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/morphology"

    def apply(self, raw_image, kernel_size):
        gray = raw_image["image"]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = _to_uint8(gray)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return (_wrap(_to_float32(closed), raw_image["bit_depth"]),)


class MorphGradient:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/morphology"

    def apply(self, raw_image, kernel_size):
        gray = raw_image["image"]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = _to_uint8(gray)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        return (_wrap(_to_float32(gradient), raw_image["bit_depth"]),)


# =============================================================================
# Adjustment Nodes
# =============================================================================

class AdjustBrightness:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "opencv/adjust"

    def adjust(self, raw_image, factor):
        gray = raw_image["image"]
        result = np.clip(gray * factor, 0, 1).astype(np.float32)
        return (_wrap(result, raw_image["bit_depth"]),)


class AdjustContrast:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "opencv/adjust"

    def adjust(self, raw_image, factor):
        gray = raw_image["image"]
        mean = np.mean(gray)
        result = np.clip((gray - mean) * factor + mean, 0, 1).astype(np.float32)
        return (_wrap(result, raw_image["bit_depth"]),)


class HistogramEqualize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "equalize"
    CATEGORY = "opencv/adjust"

    def equalize(self, raw_image):
        gray = raw_image["image"]
        img = _to_uint8(gray)
        equalized = cv2.equalizeHist(img)
        return (_wrap(_to_float32(equalized), raw_image["bit_depth"]),)


# =============================================================================
# Threshold Nodes
# =============================================================================

class Threshold:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (["binary", "binary_inv", "trunc", "tozero", "tozero_inv", "otsu"],),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/threshold"

    def apply(self, raw_image, threshold, method):
        gray = raw_image["image"]
        methods = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV,
            "otsu": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        }
        img = _to_uint8(gray)
        _, threshed = cv2.threshold(img, int(threshold * 255), 255, methods[method])
        return (_wrap(_to_float32(threshed), raw_image["bit_depth"]),)


class AdaptiveThreshold:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "block_size": ("INT", {"default": 11, "min": 3, "max": 99, "step": 2}),
                "c": ("INT", {"default": 2, "min": -50, "max": 50}),
                "method": (["mean", "gaussian"],),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/threshold"

    def apply(self, raw_image, block_size, c, method):
        gray = raw_image["image"]
        methods = {
            "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
            "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        }
        img = _to_uint8(gray)
        threshed = cv2.adaptiveThreshold(img, 255, methods[method], cv2.THRESH_BINARY, block_size, c)
        return (_wrap(_to_float32(threshed), raw_image["bit_depth"]),)


# =============================================================================
# Composite Nodes
# =============================================================================

class Blend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image1": ("RAW_IMAGE",),
                "raw_image2": ("RAW_IMAGE",),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["normal", "add", "multiply", "screen", "overlay"],),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "opencv/composite"

    def blend(self, raw_image1, raw_image2, alpha, mode):
        a = raw_image1["image"]
        b = raw_image2["image"]

        if mode == "normal":
            result = a * alpha + b * (1 - alpha)
        elif mode == "add":
            result = np.clip(a + b, 0, 1)
        elif mode == "multiply":
            result = a * b
        elif mode == "screen":
            result = 1 - (1 - a) * (1 - b)
        elif mode == "overlay":
            mask = b < 0.5
            result = np.where(mask, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        else:
            result = a * alpha + b * (1 - alpha)

        return (_wrap(np.clip(result, 0, 1).astype(np.float32), raw_image1["bit_depth"]),)


class MaskApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "mask": ("RAW_IMAGE",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "opencv/composite"

    def apply(self, raw_image, mask):
        result = raw_image["image"] * mask["image"]
        return (_wrap(result.astype(np.float32), raw_image["bit_depth"]),)


# =============================================================================
# Draw Nodes
# =============================================================================

class DrawText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "text": ("STRING", {"default": "Hello"}),
                "x": ("INT", {"default": 10, "min": 0, "max": MAX_RESOLUTION}),
                "y": ("INT", {"default": 30, "min": 0, "max": MAX_RESOLUTION}),
                "font_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "opencv/draw"

    def draw(self, raw_image, text, x, y, font_scale, intensity, thickness):
        gray = raw_image["image"]
        img = _to_uint8(gray).copy()
        val = int(intensity * 255)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, val, thickness)
        return (_wrap(_to_float32(img), raw_image["bit_depth"]),)


class DrawRectangle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "x": ("INT", {"default": 10, "min": 0, "max": MAX_RESOLUTION}),
                "y": ("INT", {"default": 10, "min": 0, "max": MAX_RESOLUTION}),
                "width": ("INT", {"default": 100, "min": 1, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 100, "min": 1, "max": MAX_RESOLUTION}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thickness": ("INT", {"default": 2, "min": -1, "max": 20}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "opencv/draw"

    def draw(self, raw_image, x, y, width, height, intensity, thickness):
        gray = raw_image["image"]
        img = _to_uint8(gray).copy()
        val = int(intensity * 255)
        cv2.rectangle(img, (x, y), (x + width, y + height), val, thickness)
        return (_wrap(_to_float32(img), raw_image["bit_depth"]),)


class DrawCircle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "center_x": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION}),
                "center_y": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION}),
                "radius": ("INT", {"default": 50, "min": 1, "max": MAX_RESOLUTION}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thickness": ("INT", {"default": 2, "min": -1, "max": 20}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "opencv/draw"

    def draw(self, raw_image, center_x, center_y, radius, intensity, thickness):
        gray = raw_image["image"]
        img = _to_uint8(gray).copy()
        val = int(intensity * 255)
        cv2.circle(img, (center_x, center_y), radius, val, thickness)
        return (_wrap(_to_float32(img), raw_image["bit_depth"]),)


class DrawLine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "x1": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "y1": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                "x2": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION}),
                "y2": ("INT", {"default": 100, "min": 0, "max": MAX_RESOLUTION}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "opencv/draw"

    def draw(self, raw_image, x1, y1, x2, y2, intensity, thickness):
        gray = raw_image["image"]
        img = _to_uint8(gray).copy()
        val = int(intensity * 255)
        cv2.line(img, (x1, y1), (x2, y2), val, thickness)
        return (_wrap(_to_float32(img), raw_image["bit_depth"]),)


# =============================================================================
# Analysis Nodes
# =============================================================================

class ImageInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("height", "width", "bit_depth")
    FUNCTION = "info"
    CATEGORY = "opencv/analysis"

    def info(self, raw_image):
        gray = raw_image["image"]
        h, w = gray.shape[:2]
        return (h, w, raw_image["bit_depth"])


class Histogram:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "bins": ("INT", {"default": 256, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "compute"
    CATEGORY = "opencv/analysis"

    def compute(self, raw_image, bins):
        gray = raw_image["image"]
        img = _to_uint8(gray)

        hist_h = 256
        hist_w = 512
        hist_img = np.zeros((hist_h, hist_w), dtype=np.uint8)

        hist = cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()
        if hist.max() > 0:
            hist = hist / hist.max() * (hist_h - 1)

        for j in range(1, bins):
            x1 = int((j - 1) * hist_w / bins)
            x2 = int(j * hist_w / bins)
            y1 = hist_h - int(hist[j - 1])
            y2 = hist_h - int(hist[j])
            cv2.line(hist_img, (x1, y1), (x2, y2), 255, 2)

        return (_wrap(_to_float32(hist_img), raw_image["bit_depth"]),)


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

    # Adjust
    "AdjustBrightness": AdjustBrightness,
    "AdjustContrast": AdjustContrast,
    "HistogramEqualize": HistogramEqualize,

    # Threshold
    "Threshold": Threshold,
    "AdaptiveThreshold": AdaptiveThreshold,

    # Composite
    "Blend": Blend,
    "MaskApply": MaskApply,

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

    # Adjust
    "AdjustBrightness": "Adjust Brightness",
    "AdjustContrast": "Adjust Contrast",
    "HistogramEqualize": "Histogram Equalize",

    # Threshold
    "Threshold": "Threshold",
    "AdaptiveThreshold": "Adaptive Threshold",

    # Composite
    "Blend": "Blend",
    "MaskApply": "Mask Apply",

    # Draw
    "DrawText": "Draw Text",
    "DrawRectangle": "Draw Rectangle",
    "DrawCircle": "Draw Circle",
    "DrawLine": "Draw Line",

    # Analysis
    "ImageInfo": "Image Info",
    "Histogram": "Histogram",
}
