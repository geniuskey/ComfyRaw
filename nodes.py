"""
ComfyRaw - Basic Image Processing Nodes
CPU-based OpenCV/numpy implementation
"""

from __future__ import annotations

import os
import json
import hashlib
import random
import logging

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

import numpy as np
import cv2

from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import comfy.model_management
from comfy.cli_args import args

import folder_paths
import node_helpers


def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()


def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)


MAX_RESOLUTION = 16384


# =============================================================================
# Image I/O Nodes
# =============================================================================

class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            image_np = np.expand_dims(image_np, 0)  # Add batch dimension

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - mask
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - mask
            else:
                mask = np.zeros((64, 64), dtype=np.float32)

            output_images.append(image_np)
            output_masks.append(np.expand_dims(mask, 0))

            if img.format == "MPO":
                break

        if len(output_images) > 1:
            output_image = np.concatenate(output_images, axis=0)
            output_mask = np.concatenate(output_masks, axis=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class LoadImageMask:
    _color_channels = ["alpha", "red", "green", "blue"]

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     "channel": (s._color_channels, ), }
                }

    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image"

    def load_image(self, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        i = node_helpers.pillow(Image.open, image_path)
        i = node_helpers.pillow(ImageOps.exif_transpose, i)

        if i.getbands() != ("R", "G", "B", "A"):
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            i = i.convert("RGBA")

        c = channel[0].upper()
        if c in i.getbands():
            mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
            if c == 'A':
                mask = 1. - mask
        else:
            mask = np.zeros((64, 64), dtype=np.float32)

        return (np.expand_dims(mask, 0),)

    @classmethod
    def IS_CHANGED(s, image, channel):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()


class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyRaw"})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyRaw", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        for (batch_number, image) in enumerate(images):
            i = (image * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(i)

            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


class PreviewImage(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }


# =============================================================================
# Image Transform Nodes
# =============================================================================

class ImageScale:
    upscale_methods = ["nearest", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "upscale_method": (s.upscale_methods,),
            "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            "crop": (s.crop_methods,)
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            return (image,)

        interp_methods = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interp = interp_methods.get(upscale_method, cv2.INTER_LINEAR)

        batch_size = image.shape[0]
        orig_height, orig_width = image.shape[1], image.shape[2]

        if width == 0:
            width = max(1, round(orig_width * height / orig_height))
        elif height == 0:
            height = max(1, round(orig_height * width / orig_width))

        results = []
        for i in range(batch_size):
            img = image[i]

            if crop == "center":
                old_aspect = orig_width / orig_height
                new_aspect = width / height

                if old_aspect > new_aspect:
                    new_width = int(orig_height * new_aspect)
                    x_start = (orig_width - new_width) // 2
                    img = img[:, x_start:x_start+new_width, :]
                else:
                    new_height = int(orig_width / new_aspect)
                    y_start = (orig_height - new_height) // 2
                    img = img[y_start:y_start+new_height, :, :]

            img_uint8 = (img * 255).astype(np.uint8)
            resized = cv2.resize(img_uint8, (width, height), interpolation=interp)
            results.append(resized.astype(np.float32) / 255.0)

        return (np.stack(results, axis=0),)


class ImageScaleBy:
    upscale_methods = ["nearest", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "upscale_method": (s.upscale_methods,),
            "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, scale_by):
        width = round(image.shape[2] * scale_by)
        height = round(image.shape[1] * scale_by)

        scale_node = ImageScale()
        return scale_node.upscale(image, upscale_method, width, height, "disabled")


class ImageInvert:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "invert"
    CATEGORY = "image"

    def invert(self, image):
        return (1.0 - image,)


class ImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",), "image2": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"
    CATEGORY = "image"

    def batch(self, image1, image2):
        if image1.shape[1:3] != image2.shape[1:3]:
            # Resize image2 to match image1
            scale_node = ImageScale()
            image2, = scale_node.upscale(image2, "bilinear", image1.shape[2], image1.shape[1], "center")

        return (np.concatenate((image1, image2), axis=0),)


class EmptyImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, color=0):
        r = ((color >> 16) & 0xFF) / 255.0
        g = ((color >> 8) & 0xFF) / 255.0
        b = (color & 0xFF) / 255.0

        image = np.full((batch_size, height, width, 3), [r, g, b], dtype=np.float32)
        return (image,)


class ImageFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "flip_method": (["horizontal", "vertical", "both"],),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flip"
    CATEGORY = "image/transform"

    def flip(self, image, flip_method):
        results = []
        for i in range(image.shape[0]):
            img = image[i]
            if flip_method == "horizontal":
                img = np.fliplr(img)
            elif flip_method == "vertical":
                img = np.flipud(img)
            else:  # both
                img = np.fliplr(np.flipud(img))
            results.append(img)
        return (np.stack(results, axis=0),)


class ImageRotate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "rotation": (["90", "180", "270"],),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate"
    CATEGORY = "image/transform"

    def rotate(self, image, rotation):
        results = []
        k = int(rotation) // 90
        for i in range(image.shape[0]):
            img = np.rot90(image[i], k=k)
            results.append(img)
        return (np.stack(results, axis=0),)


class ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            "width": ("INT", {"default": 256, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            "height": ("INT", {"default": 256, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "image/transform"

    def crop(self, image, x, y, width, height):
        results = []
        for i in range(image.shape[0]):
            cropped = image[i, y:y+height, x:x+width, :]
            results.append(cropped)
        return (np.stack(results, axis=0),)


# =============================================================================
# Node Mappings
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadImage": LoadImage,
    "LoadImageMask": LoadImageMask,
    "SaveImage": SaveImage,
    "PreviewImage": PreviewImage,
    "ImageScale": ImageScale,
    "ImageScaleBy": ImageScaleBy,
    "ImageInvert": ImageInvert,
    "ImageBatch": ImageBatch,
    "EmptyImage": EmptyImage,
    "ImageFlip": ImageFlip,
    "ImageRotate": ImageRotate,
    "ImageCrop": ImageCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImage": "Load Image",
    "LoadImageMask": "Load Image (Mask)",
    "SaveImage": "Save Image",
    "PreviewImage": "Preview Image",
    "ImageScale": "Image Scale",
    "ImageScaleBy": "Image Scale By",
    "ImageInvert": "Image Invert",
    "ImageBatch": "Image Batch",
    "EmptyImage": "Empty Image",
    "ImageFlip": "Image Flip",
    "ImageRotate": "Image Rotate",
    "ImageCrop": "Image Crop",
}


# =============================================================================
# Extra Nodes Loading
# =============================================================================

import importlib
import pkgutil
import logging

EXTENSION_WEB_DIRS = {}


def load_extra_nodes_from_package(package_name):
    """Load nodes from a package directory"""
    try:
        package = importlib.import_module(package_name)
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
            if modname.startswith("nodes_"):
                try:
                    module = importlib.import_module(f"{package_name}.{modname}")
                    if hasattr(module, "NODE_CLASS_MAPPINGS"):
                        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                    if hasattr(module, "WEB_DIRECTORY"):
                        EXTENSION_WEB_DIRS[modname] = module.WEB_DIRECTORY
                    logging.info(f"Loaded extra nodes from {package_name}.{modname}")
                except Exception as e:
                    logging.warning(f"Failed to load {package_name}.{modname}: {e}")
    except ImportError as e:
        logging.warning(f"Failed to import package {package_name}: {e}")


async def init_extra_nodes(init_custom_nodes=True, init_api_nodes=False):
    """Initialize extra nodes from comfy_extras and custom_nodes"""
    # Load comfy_extras nodes
    load_extra_nodes_from_package("comfy_extras")

    # Load custom nodes if enabled
    if init_custom_nodes:
        import os
        import sys
        custom_nodes_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_nodes")
        if os.path.isdir(custom_nodes_path):
            if custom_nodes_path not in sys.path:
                sys.path.insert(0, custom_nodes_path)
            for item in os.listdir(custom_nodes_path):
                item_path = os.path.join(custom_nodes_path, item)
                if os.path.isdir(item_path) and os.path.isfile(os.path.join(item_path, "__init__.py")):
                    try:
                        module = importlib.import_module(item)
                        if hasattr(module, "NODE_CLASS_MAPPINGS"):
                            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                        logging.info(f"Loaded custom nodes from {item}")
                    except Exception as e:
                        logging.warning(f"Failed to load custom nodes from {item}: {e}")
                elif item.endswith(".py") and item != "__init__.py":
                    modname = item[:-3]
                    try:
                        module = importlib.import_module(modname)
                        if hasattr(module, "NODE_CLASS_MAPPINGS"):
                            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                        logging.info(f"Loaded custom node {modname}")
                    except Exception as e:
                        logging.warning(f"Failed to load custom node {modname}: {e}")

    logging.info(f"Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

# For compatibility with server.py custom node manager
LOADED_MODULE_DIRS: dict = {}
