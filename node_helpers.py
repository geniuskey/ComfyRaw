"""
ComfyRaw - Node helper functions
"""

import hashlib
import numpy as np

from comfy.cli_args import args

from PIL import ImageFile, UnidentifiedImageError


def pillow(fn, arg):
    """Safe PIL image loading with truncated image support"""
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def hasher():
    """Get the configured hashing function"""
    hashfuncs = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    return hashfuncs[args.default_hashing_function]


def string_to_numpy_dtype(string):
    """Convert string to numpy dtype"""
    dtypes = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": np.float32,  # numpy doesn't have bfloat16, use float32
        "fp8_e4m3fn": np.float32,
        "fp8_e5m2": np.float32,
    }
    return dtypes.get(string, np.float32)
