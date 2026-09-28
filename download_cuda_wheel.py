"""
Resumable PyTorch CUDA 12.8 (Blackwell sm_120) Wheel Downloader & Installer.
Downloads the official PyTorch CUDA 12.8 build for NVIDIA GeForce RTX 50-series (Blackwell),
resumes interrupted downloads automatically, installs into .venv, and verifies CUDA execution.
"""

import os
import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

WHEEL_URL = "https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl"
WHEEL_FILENAME = "torch-2.11.0+cu128-cp312-cp312-win_amd64.whl"


def download_and_install():
    wheel_path = Path(WHEEL_FILENAME)
    print("=" * 60)
    print("RESUMABLE PYTORCH CUDA 12.8 (BLACKWELL sm_120) INSTALLER")
    print("=" * 60)
    print(f"Target Hardware:    NVIDIA GeForce RTX 5050 Laptop GPU (sm_120)")
    print(f"Required Package:   PyTorch 2.11+cu128 (Blackwell native kernels)")
    print(f"Destination:        {wheel_path.resolve()}")
    print("=" * 60 + "\n")

    # Use curl.exe with -C - for resumable download
    curl_cmd = [
        "curl.exe",
        "-L",
        "-C", "-",                       # Resume from where it left off
        "--retry", "50",                  # Retry up to 50 times on connection drops
        "--retry-delay", "5",
        "--retry-max-time", "14400",      # Up to 4 hours total retry window
        "--connect-timeout", "30",
        "-o", str(wheel_path),
        WHEEL_URL
    ]

    print("[INFO] Starting / Resuming wheel download via curl...")
    ret = subprocess.run(curl_cmd)
    if ret.returncode != 0:
        print(f"[ERROR] Download failed with code {ret.returncode}. Re-run this script when connection is stable to resume.")
        return False

    print("\n[INFO] Download complete! Installing into local .venv environment...")
    install_cmd = [
        sys.executable,
        "-m", "uv", "pip", "install",
        "--python", str(Path(".venv/Scripts/python.exe")),
        "--reinstall",
        str(wheel_path)
    ]
    ret_install = subprocess.run(install_cmd)
    if ret_install.returncode != 0:
        print("[ERROR] Installation into .venv failed.")
        return False

    print("\n[INFO] Verifying GPU execution on RTX 5050...")
    verify_cmd = [str(Path(".venv/Scripts/python.exe")), "check_cuda.py"]
    subprocess.run(verify_cmd)
    return True


if __name__ == "__main__":
    download_and_install()
