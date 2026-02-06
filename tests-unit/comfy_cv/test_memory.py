"""
ComfyRaw - Unit tests for comfy_cv.memory module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
import numpy as np
from comfy_cv.memory import MemoryManager


class TestMemoryInfo:
    """Tests for memory information functions"""

    def test_get_memory_info(self):
        """Get memory info dictionary"""
        info = MemoryManager.get_memory_info()

        assert "total" in info
        assert "available" in info
        assert "used" in info
        assert "percent" in info
        assert "free" in info

        assert info["total"] > 0
        assert info["available"] >= 0
        assert info["used"] >= 0
        assert 0 <= info["percent"] <= 100

    def test_get_available_memory(self):
        """Get available memory"""
        available = MemoryManager.get_available_memory()

        assert isinstance(available, int)
        assert available > 0

    def test_get_total_memory(self):
        """Get total memory"""
        total = MemoryManager.get_total_memory()

        assert isinstance(total, int)
        assert total > 0

    def test_get_used_memory(self):
        """Get used memory"""
        used = MemoryManager.get_used_memory()

        assert isinstance(used, int)
        assert used >= 0

    def test_get_memory_percent(self):
        """Get memory usage percentage"""
        percent = MemoryManager.get_memory_percent()

        assert isinstance(percent, float)
        assert 0 <= percent <= 100


class TestMemoryEstimation:
    """Tests for memory estimation functions"""

    def test_estimate_image_memory(self):
        """Estimate memory for image shape"""
        # 1920x1080 RGB float32 image
        shape = (1, 1080, 1920, 3)
        estimate = MemoryManager.estimate_image_memory(shape, dtype=np.float32)

        # Expected: 1 * 1080 * 1920 * 3 * 4 bytes = 24,883,200 bytes
        expected = 1 * 1080 * 1920 * 3 * 4
        assert estimate == expected

    def test_estimate_image_memory_uint8(self):
        """Estimate memory for uint8 image"""
        shape = (1, 1080, 1920, 3)
        estimate = MemoryManager.estimate_image_memory(shape, dtype=np.uint8)

        # Expected: 1 * 1080 * 1920 * 3 * 1 byte = 6,220,800 bytes
        expected = 1 * 1080 * 1920 * 3 * 1
        assert estimate == expected

    def test_estimate_batch_memory(self):
        """Estimate memory for batch"""
        estimate = MemoryManager.estimate_batch_memory(
            batch_size=4, height=512, width=512, channels=3, dtype=np.float32
        )

        # Expected: 4 * 512 * 512 * 3 * 4 bytes = 12,582,912 bytes
        expected = 4 * 512 * 512 * 3 * 4
        assert estimate == expected


class TestMemoryChecks:
    """Tests for memory check functions"""

    def test_should_free_memory(self):
        """Check if memory should be freed"""
        result = MemoryManager.should_free_memory()

        assert isinstance(result, bool)

    def test_is_memory_warning(self):
        """Check if memory warning threshold reached"""
        result = MemoryManager.is_memory_warning()

        assert isinstance(result, bool)

    def test_can_allocate_small(self):
        """Check if small allocation is possible"""
        # 1 MB should always be allocatable
        result = MemoryManager.can_allocate(1024 * 1024)

        assert result is True

    def test_can_allocate_huge(self):
        """Check if huge allocation is denied"""
        # 1 TB should not be allocatable
        result = MemoryManager.can_allocate(1024 * 1024 * 1024 * 1024)

        assert result is False


class TestFreeMemory:
    """Tests for free_memory function"""

    def test_free_memory(self):
        """Run garbage collection"""
        # Create some garbage
        _ = [np.zeros((1000, 1000)) for _ in range(10)]

        freed = MemoryManager.free_memory()

        assert isinstance(freed, int)
        assert freed >= 0


class TestFormatBytes:
    """Tests for format_bytes function"""

    def test_format_bytes(self):
        """Format bytes to human readable string"""
        assert MemoryManager.format_bytes(0) == "0.0 B"
        assert MemoryManager.format_bytes(1023) == "1023.0 B"
        assert MemoryManager.format_bytes(1024) == "1.0 KB"
        assert MemoryManager.format_bytes(1024 * 1024) == "1.0 MB"
        assert MemoryManager.format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert MemoryManager.format_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"

    def test_format_bytes_decimal(self):
        """Format bytes with decimals"""
        assert "1.5" in MemoryManager.format_bytes(int(1.5 * 1024 * 1024))


class TestMemorySummary:
    """Tests for memory_summary function"""

    def test_memory_summary(self):
        """Get memory summary string"""
        summary = MemoryManager.memory_summary()

        assert isinstance(summary, str)
        assert "Memory:" in summary
        assert "used" in summary
        assert "total" in summary
        assert "%" in summary
