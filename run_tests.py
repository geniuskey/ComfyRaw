#!/usr/bin/env python
"""
ComfyRaw - Test runner script
Run: python run_tests.py
"""

import sys
import subprocess


def main():
    """Run all tests with coverage"""
    args = [
        sys.executable, "-m", "pytest",
        "tests-unit/comfy_cv/",
        "tests-unit/nodes/",
        "-v",
        "--cov=comfy_cv",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_report",
    ]

    # Add any extra arguments passed to the script
    args.extend(sys.argv[1:])

    print("Running ComfyRaw tests...")
    print(f"Command: {' '.join(args)}")
    print("-" * 60)

    result = subprocess.run(args)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
