import numpy as np
from typing import TypedDict, Optional

ImageInput = np.ndarray
"""
An image in format [B, H, W, C] where B is the batch size, C is the number of channels,
"""

MaskInput = np.ndarray
"""
A mask in format [B, H, W] where B is the batch size
"""

class AudioInput(TypedDict):
    """
    TypedDict representing audio input.
    """

    waveform: np.ndarray
    """
    Array in the format [B, C, T] where B is the batch size, C is the number of channels,
    """

    sample_rate: int

class LatentInput(TypedDict):
    """
    TypedDict representing latent input.
    """

    samples: np.ndarray
    """
    Array in the format [B, C, H, W] where B is the batch size, C is the number of channels,
    H is the height, and W is the width.
    """

    noise_mask: Optional[MaskInput]
    """
    Optional noise mask array in the same format as samples.
    """

    batch_index: Optional[list[int]]
