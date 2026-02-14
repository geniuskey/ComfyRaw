import os
import hashlib
import math
import random
import numpy as np
import cv2
from PIL import Image
import folder_paths


WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "raw_web")

RAW_EXTENSIONS = {".raw", ".bin", ".dat"}
BIT_DEPTHS = ["8", "10", "12", "14", "16"]
BIT_DEPTH_INFO = {
    "8":  {"np_dtype": np.uint8,  "max_val": 255},
    "10": {"np_dtype": np.uint16, "max_val": 1023},
    "12": {"np_dtype": np.uint16, "max_val": 4095},
    "14": {"np_dtype": np.uint16, "max_val": 16383},
    "16": {"np_dtype": np.uint16, "max_val": 65535},
}


def _list_raw_files():
    input_dir = folder_paths.get_input_directory()
    files = []
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)):
            ext = os.path.splitext(f)[1].lower()
            if ext in RAW_EXTENSIONS:
                files.append(f)
    return sorted(files)


def _list_raw_folders():
    input_dir = folder_paths.get_input_directory()
    folders = []
    for d in os.listdir(input_dir):
        full_path = os.path.join(input_dir, d)
        if os.path.isdir(full_path):
            has_raw = any(
                os.path.splitext(f)[1].lower() in RAW_EXTENSIONS
                for f in os.listdir(full_path)
                if os.path.isfile(os.path.join(full_path, f))
            )
            if has_raw:
                folders.append(d)
    return sorted(folders)


def _auto_detect_dimensions(total_pixels):
    ratios = [(4, 3), (1, 1), (3, 2), (16, 9)]
    for rw, rh in ratios:
        w = int(math.sqrt(total_pixels * rw / rh))
        if w > 0 and total_pixels % w == 0:
            h = total_pixels // w
            return w, h

    best_w = int(math.sqrt(total_pixels))
    for offset in range(best_w):
        w = best_w - offset
        if w > 0 and total_pixels % w == 0:
            h = total_pixels // w
            return w, h

    return total_pixels, 1


def _resolve_dimensions(data_len, width, height):
    if width > 0 and height > 0:
        expected = width * height
    elif width > 0:
        height = math.ceil(data_len / width)
        expected = width * height
    elif height > 0:
        width = math.ceil(data_len / height)
        expected = width * height
    else:
        width, height = _auto_detect_dimensions(data_len)
        expected = width * height

    return width, height, expected


def _raw_to_preview_png(gray, temp_dir, prefix):
    preview_img = (gray * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(preview_img, mode='L')
    os.makedirs(temp_dir, exist_ok=True)
    preview_file = f"{prefix}_{random.randint(0, 0xFFFFFFFF):08x}.png"
    img.save(os.path.join(temp_dir, preview_file), compress_level=1)
    return preview_file


class LoadRawImage:
    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))

    @classmethod
    def INPUT_TYPES(s):
        files = _list_raw_files()
        if not files:
            files = [""]
        return {
            "required": {
                "file": (files, {"image_upload": True}),
                "bit_depth": (BIT_DEPTHS, {"default": "16"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 65536}),
                "height": ("INT", {"default": 0, "min": 0, "max": 65536}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "load_raw"
    OUTPUT_NODE = True
    CATEGORY = "raw_image"

    def load_raw(self, file, bit_depth, width, height):
        image_path = folder_paths.get_annotated_filepath(file)
        info = BIT_DEPTH_INFO[bit_depth]
        data = np.fromfile(image_path, dtype=info["np_dtype"])

        total_pixels = len(data)
        width, height, expected = _resolve_dimensions(total_pixels, width, height)

        if total_pixels < expected:
            data = np.pad(data, (0, expected - total_pixels))
        elif total_pixels > expected:
            data = data[:expected]

        data = data.reshape(height, width)
        gray = data.astype(np.float32) / info["max_val"]
        gray = gray.clip(0.0, 1.0)

        raw_image = {"image": gray, "bit_depth": int(bit_depth)}

        preview_file = _raw_to_preview_png(gray, self.temp_dir, f"LoadRawImage{self.prefix_append}")

        return {
            "ui": {
                "images": [{"filename": preview_file, "subfolder": "", "type": "temp"}],
                "text": (f"{width} x {height}",),
                "resolved_size": (f"{width} x {height}",),
            },
            "result": (raw_image,),
        }

    @classmethod
    def IS_CHANGED(s, file, bit_depth, width, height):
        image_path = folder_paths.get_annotated_filepath(file)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        m.update(bit_depth.encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, file, bit_depth, width, height):
        if not folder_paths.exists_annotated_filepath(file):
            return "Invalid raw file: {}".format(file)
        return True


class PreviewRawImage:
    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "raw_image"

    def preview(self, raw_image):
        gray = raw_image["image"]
        preview_file = _raw_to_preview_png(gray, self.temp_dir, f"PreviewRawImage{self.prefix_append}")
        return {"ui": {"images": [{"filename": preview_file, "subfolder": "", "type": "temp"}]}}


class SaveRawImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyRaw"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_raw"
    OUTPUT_NODE = True
    CATEGORY = "raw_image"

    def save_raw(self, raw_image, filename_prefix):
        gray = raw_image["image"]
        bd = raw_image["bit_depth"]
        info = BIT_DEPTH_INFO[str(bd)]

        h, w = gray.shape[:2]
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, w, h)

        raw_data = (gray.clip(0.0, 1.0) * info["max_val"]).clip(0, info["max_val"]).astype(info["np_dtype"])

        filename_with_batch = filename.replace("%batch_num%", "0")
        file = f"{filename_with_batch}_{counter:05}_.raw"
        filepath = os.path.join(full_output_folder, file)
        raw_data.tofile(filepath)

        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": self.type}]}}


class LoadRawImages:
    def __init__(self):
        self.temp_dir = folder_paths.get_temp_directory()
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))

    @classmethod
    def INPUT_TYPES(s):
        folders = _list_raw_folders()
        if not folders:
            folders = [""]
        return {
            "required": {
                "folder": (folders,),
                "bit_depth": (BIT_DEPTHS, {"default": "16"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 65536}),
                "height": ("INT", {"default": 0, "min": 0, "max": 65536}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGES",)
    FUNCTION = "load_raw"
    OUTPUT_NODE = True
    CATEGORY = "raw_image"

    def load_raw(self, folder, bit_depth, width, height):
        input_dir = folder_paths.get_input_directory()
        folder_path = os.path.join(input_dir, folder)

        files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and os.path.splitext(f)[1].lower() in RAW_EXTENSIONS
        ])

        if not files:
            raise ValueError(f"No raw files found in {folder}")

        info = BIT_DEPTH_INFO[bit_depth]
        frames = []
        resolved_w, resolved_h = width, height

        for i, fname in enumerate(files):
            fpath = os.path.join(folder_path, fname)
            data = np.fromfile(fpath, dtype=info["np_dtype"])
            total_pixels = len(data)

            if i == 0:
                resolved_w, resolved_h, expected = _resolve_dimensions(total_pixels, width, height)
            else:
                expected = resolved_w * resolved_h

            if total_pixels < expected:
                data = np.pad(data, (0, expected - total_pixels))
            elif total_pixels > expected:
                data = data[:expected]

            data = data.reshape(resolved_h, resolved_w)
            gray = data.astype(np.float32) / info["max_val"]
            gray = gray.clip(0.0, 1.0)
            frames.append(gray)

        images = np.stack(frames, axis=0)
        raw_images = {"images": images, "bit_depth": int(bit_depth)}

        preview_file = _raw_to_preview_png(frames[0], self.temp_dir, f"LoadRawImages{self.prefix_append}")

        return {
            "ui": {
                "images": [{"filename": preview_file, "subfolder": "", "type": "temp"}],
                "text": (f"{len(files)} frames, {resolved_w} x {resolved_h}",),
            },
            "result": (raw_images,),
        }

    @classmethod
    def IS_CHANGED(s, folder, bit_depth, width, height):
        input_dir = folder_paths.get_input_directory()
        folder_path = os.path.join(input_dir, folder)
        m = hashlib.sha256()
        for f in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, f)
            if os.path.isfile(fpath) and os.path.splitext(f)[1].lower() in RAW_EXTENSIONS:
                m.update(f.encode())
                m.update(str(os.path.getmtime(fpath)).encode())
        m.update(bit_depth.encode())
        return m.digest().hex()


# =============================================================================
# Statistics Nodes (RAW_IMAGES -> RAW_IMAGE)
# =============================================================================

class RawAvgFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_images": ("RAW_IMAGES",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "compute"
    CATEGORY = "raw_image/stats"

    def compute(self, raw_images):
        result = np.mean(raw_images["images"], axis=0).astype(np.float32)
        return (_wrap(result, raw_images["bit_depth"]),)


class RawStdFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_images": ("RAW_IMAGES",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "compute"
    CATEGORY = "raw_image/stats"

    def compute(self, raw_images):
        result = np.std(raw_images["images"], axis=0).astype(np.float32)
        return (_wrap(result, raw_images["bit_depth"]),)


class RawVarFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_images": ("RAW_IMAGES",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "compute"
    CATEGORY = "raw_image/stats"

    def compute(self, raw_images):
        result = np.var(raw_images["images"], axis=0).astype(np.float32)
        return (_wrap(result, raw_images["bit_depth"]),)


def _wrap(image, bit_depth):
    return {"image": image.clip(0.0, 1.0).astype(np.float32), "bit_depth": bit_depth}


# =============================================================================
# Arithmetic Nodes
# =============================================================================

class RawAdd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image1": ("RAW_IMAGE",),
                "raw_image2": ("RAW_IMAGE",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "raw_image/math"

    def apply(self, raw_image1, raw_image2):
        result = raw_image1["image"] + raw_image2["image"]
        return (_wrap(result, raw_image1["bit_depth"]),)


class RawSubtract:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image1": ("RAW_IMAGE",),
                "raw_image2": ("RAW_IMAGE",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "raw_image/math"

    def apply(self, raw_image1, raw_image2):
        result = raw_image1["image"] - raw_image2["image"]
        return (_wrap(result, raw_image1["bit_depth"]),)


class RawMultiply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image1": ("RAW_IMAGE",),
                "raw_image2": ("RAW_IMAGE",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "raw_image/math"

    def apply(self, raw_image1, raw_image2):
        result = raw_image1["image"] * raw_image2["image"]
        return (_wrap(result, raw_image1["bit_depth"]),)


class RawDivide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image1": ("RAW_IMAGE",),
                "raw_image2": ("RAW_IMAGE",),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "raw_image/math"

    def apply(self, raw_image1, raw_image2):
        divisor = raw_image2["image"]
        safe_divisor = np.where(divisor == 0, 1.0, divisor)
        result = raw_image1["image"] / safe_divisor
        return (_wrap(result, raw_image1["bit_depth"]),)


# =============================================================================
# Crop Nodes
# =============================================================================

class RawEdgeCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "top": ("INT", {"default": 0, "min": 0, "max": 65536}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 65536}),
                "left": ("INT", {"default": 0, "min": 0, "max": 65536}),
                "right": ("INT", {"default": 0, "min": 0, "max": 65536}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "raw_image/transform"

    def apply(self, raw_image, top, bottom, left, right):
        gray = raw_image["image"]
        h, w = gray.shape[:2]
        y2 = h - bottom if bottom > 0 else h
        x2 = w - right if right > 0 else w
        result = gray[top:y2, left:x2]
        return (_wrap(result, raw_image["bit_depth"]),)


class RawCenterCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "width": ("INT", {"default": 100, "min": 1, "max": 65536}),
                "height": ("INT", {"default": 100, "min": 1, "max": 65536}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "raw_image/transform"

    def apply(self, raw_image, width, height):
        gray = raw_image["image"]
        h, w = gray.shape[:2]
        crop_w = min(width, w)
        crop_h = min(height, h)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        result = gray[y:y + crop_h, x:x + crop_w]
        return (_wrap(result, raw_image["bit_depth"]),)


# =============================================================================
# Flip Node
# =============================================================================

class RawFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "direction": (["h", "v", "hv"],),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "raw_image/transform"

    def apply(self, raw_image, direction):
        gray = raw_image["image"]
        if direction == "h":
            result = np.fliplr(gray)
        elif direction == "v":
            result = np.flipud(gray)
        else:
            result = np.flipud(np.fliplr(gray))
        return (_wrap(result.copy(), raw_image["bit_depth"]),)


# =============================================================================
# Spots Helpers
# =============================================================================

OPS = {
    ">":  np.greater,
    ">=": np.greater_equal,
    "<":  np.less,
    "<=": np.less_equal,
}

OPERATORS = [">", ">=", "<", "<="]


def _apply_condition(gray, operator, threshold):
    return OPS[operator](gray, threshold)


def _is_max_op(operator):
    return operator in (">", ">=")


# =============================================================================
# Spots Nodes
# =============================================================================

class RawFindSingleSpot:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "operator": (OPERATORS,),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("x", "y", "value")
    FUNCTION = "find"
    CATEGORY = "raw_image/spots"

    def find(self, raw_image, operator, threshold):
        gray = raw_image["image"]
        mask = _apply_condition(gray, operator, threshold)

        if not np.any(mask):
            return (0, 0, 0.0)

        masked = np.where(mask, gray, -np.inf if _is_max_op(operator) else np.inf)

        if _is_max_op(operator):
            idx = np.argmax(masked)
        else:
            idx = np.argmin(masked)

        y, x = np.unravel_index(idx, gray.shape)
        value = float(gray[y, x])
        return (int(x), int(y), value)


class RawFindSpotsGrid:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "operator": (OPERATORS,),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "kernel_w": ("INT", {"default": 32, "min": 1, "max": 65536}),
                "kernel_h": ("INT", {"default": 32, "min": 1, "max": 65536}),
                "stride_w": ("INT", {"default": 32, "min": 1, "max": 65536}),
                "stride_h": ("INT", {"default": 32, "min": 1, "max": 65536}),
            }
        }

    RETURN_TYPES = ("SPOTS",)
    FUNCTION = "find"
    CATEGORY = "raw_image/spots"

    def find(self, raw_image, operator, threshold, kernel_w, kernel_h, stride_w, stride_h):
        gray = raw_image["image"]
        h, w = gray.shape
        use_max = _is_max_op(operator)
        spots = []

        for y0 in range(0, h, stride_h):
            for x0 in range(0, w, stride_w):
                y1 = min(y0 + kernel_h, h)
                x1 = min(x0 + kernel_w, w)
                window = gray[y0:y1, x0:x1]
                mask = _apply_condition(window, operator, threshold)

                if not np.any(mask):
                    continue

                masked = np.where(mask, window, -np.inf if use_max else np.inf)
                idx = np.argmax(masked) if use_max else np.argmin(masked)
                wy, wx = np.unravel_index(idx, window.shape)
                spots.append({"x": int(x0 + wx), "y": int(y0 + wy), "value": float(window[wy, wx])})

        return (spots,)


class RawFindSpotsGroup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "operator": (OPERATORS,),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
                "min_area": ("INT", {"default": 1, "min": 1, "max": 1000000}),
                "max_area": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "point_mode": (["peak", "centroid"],),
            }
        }

    RETURN_TYPES = ("SPOTS",)
    FUNCTION = "find"
    CATEGORY = "raw_image/spots"

    def find(self, raw_image, operator, threshold, min_area, max_area, point_mode):
        gray = raw_image["image"]
        mask = _apply_condition(gray, operator, threshold).astype(np.uint8)
        use_max = _is_max_op(operator)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        spots = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            if max_area > 0 and area > max_area:
                continue

            component_mask = labels == i

            if point_mode == "centroid":
                cx, cy = centroids[i]
                x, y = int(round(cx)), int(round(cy))
                y = min(y, gray.shape[0] - 1)
                x = min(x, gray.shape[1] - 1)
                value = float(gray[y, x])
            else:
                component_vals = np.where(component_mask, gray, -np.inf if use_max else np.inf)
                idx = np.argmax(component_vals) if use_max else np.argmin(component_vals)
                y, x = np.unravel_index(idx, gray.shape)
                value = float(gray[y, x])

            spots.append({"x": int(x), "y": int(y), "value": value})

        return (spots,)


class RawDrawSpots:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "spots": ("SPOTS",),
                "marker_size": ("INT", {"default": 5, "min": 1, "max": 100}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "raw_image/spots"

    def draw(self, raw_image, spots, marker_size, intensity):
        gray = raw_image["image"].copy()
        for spot in spots:
            cv2.circle(gray, (spot["x"], spot["y"]), marker_size, float(intensity), -1)
        return (_wrap(gray, raw_image["bit_depth"]),)


class RawSpotsInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "spots": ("SPOTS",),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("count", "summary")
    FUNCTION = "info"
    CATEGORY = "raw_image/spots"

    def info(self, spots):
        count = len(spots)
        lines = [f"Total spots: {count}"]
        for i, s in enumerate(spots):
            lines.append(f"  [{i}] x={s['x']}, y={s['y']}, value={s['value']:.6f}")
        summary = "\n".join(lines)
        return (count, summary)


# =============================================================================
# Pattern Channel Extraction Nodes
# =============================================================================

class RawExtractChannel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "cell_w": ("INT", {"default": 2, "min": 1, "max": 16}),
                "cell_h": ("INT", {"default": 2, "min": 1, "max": 16}),
                "offset_x": ("INT", {"default": 0, "min": 0, "max": 15}),
                "offset_y": ("INT", {"default": 0, "min": 0, "max": 15}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "extract"
    CATEGORY = "raw_image/pattern"

    def extract(self, raw_image, cell_w, cell_h, offset_x, offset_y):
        gray = raw_image["image"]
        h, w = gray.shape[:2]
        cropped = gray[:h // cell_h * cell_h, :w // cell_w * cell_w]
        result = cropped[offset_y::cell_h, offset_x::cell_w]
        return (_wrap(result.copy(), raw_image["bit_depth"]),)


class RawExtractAllChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_image": ("RAW_IMAGE",),
                "cell_w": ("INT", {"default": 2, "min": 1, "max": 16}),
                "cell_h": ("INT", {"default": 2, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGES",)
    FUNCTION = "extract"
    CATEGORY = "raw_image/pattern"

    def extract(self, raw_image, cell_w, cell_h):
        gray = raw_image["image"]
        h, w = gray.shape[:2]
        cropped = gray[:h // cell_h * cell_h, :w // cell_w * cell_w]
        frames = []
        for oy in range(cell_h):
            for ox in range(cell_w):
                frames.append(cropped[oy::cell_h, ox::cell_w])
        images = np.stack(frames, axis=0)
        return ({"images": images, "bit_depth": raw_image["bit_depth"]},)


# =============================================================================
# Frame Selection Nodes
# =============================================================================

class RawSelectFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_images": ("RAW_IMAGES",),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("RAW_IMAGE",)
    FUNCTION = "select"
    CATEGORY = "raw_image"

    def select(self, raw_images, index):
        images = raw_images["images"]
        idx = min(index, images.shape[0] - 1)
        return (_wrap(images[idx].copy(), raw_images["bit_depth"]),)


class RawFrameCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_images": ("RAW_IMAGES",),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count"
    CATEGORY = "raw_image"

    def count(self, raw_images):
        return (raw_images["images"].shape[0],)


NODE_CLASS_MAPPINGS = {
    "LoadRawImage": LoadRawImage,
    "LoadRawImages": LoadRawImages,
    "PreviewRawImage": PreviewRawImage,
    "SaveRawImage": SaveRawImage,
    "RawAvgFrame": RawAvgFrame,
    "RawStdFrame": RawStdFrame,
    "RawVarFrame": RawVarFrame,
    "RawAdd": RawAdd,
    "RawSubtract": RawSubtract,
    "RawMultiply": RawMultiply,
    "RawDivide": RawDivide,
    "RawEdgeCrop": RawEdgeCrop,
    "RawCenterCrop": RawCenterCrop,
    "RawFlip": RawFlip,
    "RawFindSingleSpot": RawFindSingleSpot,
    "RawFindSpotsGrid": RawFindSpotsGrid,
    "RawFindSpotsGroup": RawFindSpotsGroup,
    "RawDrawSpots": RawDrawSpots,
    "RawSpotsInfo": RawSpotsInfo,
    "RawExtractChannel": RawExtractChannel,
    "RawExtractAllChannels": RawExtractAllChannels,
    "RawSelectFrame": RawSelectFrame,
    "RawFrameCount": RawFrameCount,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRawImage": "Load Raw Image",
    "LoadRawImages": "Load Raw Images",
    "PreviewRawImage": "Preview Raw Image",
    "SaveRawImage": "Save Raw Image",
    "RawAvgFrame": "Raw Avg Frame",
    "RawStdFrame": "Raw Std Frame",
    "RawVarFrame": "Raw Var Frame",
    "RawAdd": "Raw Add",
    "RawSubtract": "Raw Subtract",
    "RawMultiply": "Raw Multiply",
    "RawDivide": "Raw Divide",
    "RawEdgeCrop": "Raw Edge Crop",
    "RawCenterCrop": "Raw Center Crop",
    "RawFlip": "Raw Flip",
    "RawFindSingleSpot": "Raw Find Single Spot",
    "RawFindSpotsGrid": "Raw Find Spots Grid",
    "RawFindSpotsGroup": "Raw Find Spots Group",
    "RawDrawSpots": "Raw Draw Spots",
    "RawSpotsInfo": "Raw Spots Info",
    "RawExtractChannel": "Raw Extract Channel",
    "RawExtractAllChannels": "Raw Extract All Channels",
    "RawSelectFrame": "Raw Select Frame",
    "RawFrameCount": "Raw Frame Count",
}
