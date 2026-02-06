import os
import hashlib
import math
import random
import numpy as np
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


NODE_CLASS_MAPPINGS = {
    "LoadRawImage": LoadRawImage,
    "PreviewRawImage": PreviewRawImage,
    "SaveRawImage": SaveRawImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRawImage": "Load Raw Image",
    "PreviewRawImage": "Preview Raw Image",
    "SaveRawImage": "Save Raw Image",
}
