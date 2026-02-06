"""
ComfyRaw - Root test configuration
"""

import sys
import os

# Add project root to Python path before anything else
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Force reimport to ensure comfy_cv is found
if 'comfy_cv' in sys.modules:
    del sys.modules['comfy_cv']

import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs"""
    return str(tmp_path)
