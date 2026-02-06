"""
ComfyRaw - Memory management utilities
"""

import gc
import psutil
import numpy as np
from typing import Tuple


class MemoryManager:
    """Memory management for CPU-based image processing"""

    # Minimum free memory threshold (512 MB)
    MIN_FREE_MEMORY = 512 * 1024 * 1024

    # Warning threshold (1 GB)
    WARNING_THRESHOLD = 1024 * 1024 * 1024

    @staticmethod
    def get_memory_info() -> dict:
        """Get system memory information"""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent,
            "free": mem.free,
        }

    @staticmethod
    def get_available_memory() -> int:
        """Get available system RAM in bytes"""
        return psutil.virtual_memory().available

    @staticmethod
    def get_total_memory() -> int:
        """Get total system RAM in bytes"""
        return psutil.virtual_memory().total

    @staticmethod
    def get_used_memory() -> int:
        """Get used system RAM in bytes"""
        return psutil.virtual_memory().used

    @staticmethod
    def get_memory_percent() -> float:
        """Get memory usage percentage"""
        return psutil.virtual_memory().percent

    @staticmethod
    def estimate_image_memory(shape: Tuple, dtype=np.float32) -> int:
        """Estimate memory needed for an image array"""
        element_size = np.dtype(dtype).itemsize
        return int(np.prod(shape) * element_size)

    @staticmethod
    def estimate_batch_memory(batch_size: int, height: int, width: int,
                               channels: int = 3, dtype=np.float32) -> int:
        """Estimate memory for a batch of images"""
        return MemoryManager.estimate_image_memory(
            (batch_size, height, width, channels), dtype
        )

    @staticmethod
    def should_free_memory() -> bool:
        """Check if memory cleanup is recommended"""
        return MemoryManager.get_available_memory() < MemoryManager.MIN_FREE_MEMORY

    @staticmethod
    def is_memory_warning() -> bool:
        """Check if memory is getting low"""
        return MemoryManager.get_available_memory() < MemoryManager.WARNING_THRESHOLD

    @staticmethod
    def free_memory(force: bool = False) -> int:
        """Run garbage collection and return freed memory estimate"""
        before = MemoryManager.get_used_memory()
        gc.collect()
        after = MemoryManager.get_used_memory()
        return max(0, before - after)

    @staticmethod
    def can_allocate(size: int, safety_margin: float = 0.1) -> bool:
        """Check if we can safely allocate the given size"""
        available = MemoryManager.get_available_memory()
        required = size + int(available * safety_margin)
        return available > required

    @staticmethod
    def format_bytes(size: int) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(size) < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    @staticmethod
    def memory_summary() -> str:
        """Get memory usage summary string"""
        info = MemoryManager.get_memory_info()
        return (
            f"Memory: {MemoryManager.format_bytes(info['used'])} used / "
            f"{MemoryManager.format_bytes(info['total'])} total "
            f"({info['percent']:.1f}%)"
        )
