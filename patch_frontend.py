"""Patch comfyui-frontend-package to use ComfyRaw default workflow.

Run after `uv sync` or when the frontend package is updated:
    uv run python patch_frontend.py
"""
import re
import shutil
from pathlib import Path

FRONTEND_ASSETS = Path(__file__).parent / ".venv" / "Lib" / "site-packages" / "comfyui_frontend_package" / "static" / "assets"
LOCAL_ASSETS = Path(__file__).parent / "web" / "assets"

def patch():
    patched = 0
    for local_file in LOCAL_ASSETS.glob("*.js"):
        target = FRONTEND_ASSETS / local_file.name
        if target.exists():
            shutil.copy2(local_file, target)
            print(f"Patched: {local_file.name}")
            patched += 1
        else:
            print(f"Skipped (not in frontend): {local_file.name}")

    if patched == 0:
        print("No files to patch. Frontend package version may have changed.")
        print("Run: grep -l 'KSampler' .venv/.../assets/*.js to find the new target file.")
    else:
        print(f"\nDone: {patched} file(s) patched.")

if __name__ == "__main__":
    patch()
