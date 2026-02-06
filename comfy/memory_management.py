"""
ComfyRaw - Memory management utilities
Simplified version for numpy-based image processing
"""

import math
import numpy as np


def array_memory_size(arr):
    """Calculate memory size of numpy array in bytes"""
    if arr is None:
        return 0
    if isinstance(arr, np.ndarray):
        return arr.nbytes
    if isinstance(arr, list):
        return sum(array_memory_size(a) for a in arr)
    return 0


def estimate_batch_memory(shape, dtype=np.float32):
    """Estimate memory needed for a batch of images"""
    element_size = np.dtype(dtype).itemsize
    return math.prod(shape) * element_size
