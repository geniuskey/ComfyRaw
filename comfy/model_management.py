"""
ComfyRaw - CPU-based memory management
Simplified version for OpenCV image processing (no GPU/torch dependencies)
"""

import psutil
import logging
import threading
import gc

# Memory thresholds
MIN_FREE_MEMORY = 512 * 1024 * 1024  # 512 MB minimum free RAM


def get_free_memory():
    """Get available system RAM in bytes"""
    return psutil.virtual_memory().available


def get_total_memory():
    """Get total system RAM in bytes"""
    return psutil.virtual_memory().total


def get_memory_info():
    """Get memory usage information"""
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "used": mem.used,
        "percent": mem.percent
    }


def should_free_memory():
    """Check if memory cleanup is needed"""
    return get_free_memory() < MIN_FREE_MEMORY


def soft_empty_cache(force=False):
    """Run garbage collection to free memory"""
    if force or should_free_memory():
        gc.collect()


def cleanup():
    """Cleanup and free memory"""
    gc.collect()


def unload_all_models():
    """Placeholder for compatibility - just run garbage collection"""
    gc.collect()


def debug_memory_summary():
    """Return memory usage summary"""
    mem = psutil.virtual_memory()
    return f"RAM: {mem.used / (1024**3):.1f}GB used / {mem.total / (1024**3):.1f}GB total ({mem.percent}%)"


# Exception for out of memory (compatibility)
OOM_EXCEPTION = MemoryError


# Interrupt handling
class InterruptProcessingException(Exception):
    pass


interrupt_processing_mutex = threading.RLock()
interrupt_processing = False


def interrupt_current_processing(value=True):
    global interrupt_processing
    with interrupt_processing_mutex:
        interrupt_processing = value


def processing_interrupted():
    global interrupt_processing
    with interrupt_processing_mutex:
        return interrupt_processing


def throw_exception_if_processing_interrupted():
    global interrupt_processing
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()


# Logging
logging.info(f"ComfyRaw: Total RAM {get_total_memory() / (1024*1024):.0f} MB")
