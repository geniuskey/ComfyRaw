"""
ComfyRaw - Type definitions for node system
"""

import numpy as np
from typing import Callable, Protocol, TypedDict, Optional, List

from .node_typing import IO, InputTypeDict, ComfyNodeABC, CheckLazyMixin, FileLocator


# Image type alias
ImageArray = np.ndarray  # (B, H, W, C) float32


__all__ = [
    "ImageArray",
    IO.__name__,
    InputTypeDict.__name__,
    ComfyNodeABC.__name__,
    CheckLazyMixin.__name__,
    FileLocator.__name__,
]
