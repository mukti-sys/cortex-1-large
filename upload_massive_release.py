"""
High-Speed Mass Upload Engine for Cortex-1 Large (421M).
Leverages Rust-based hf_transfer for parallel multi-part chunked streaming.
Uploads the entire hf_release_bundle/ in a single atomic release commit.
"""

import os
import sys

# Enable Rust multi-part binary transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "mukti-sys/cortex-1-large"
BUNDLE_DIR = Path(__file__).resolve().parent / "hf_release_bundle"


def upload_release(token: str = None):
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF token missing. Set HF_TOKEN environment variable.")
        return False

    api = HfApi(token=token)
    print(f"[INFO] Authenticated as: {api.whoami().get('name')}")
    print(f"[INFO] Uploading unified release bundle from: {BUNDLE_DIR}")
    print(f"[INFO] Target repository: https://huggingface.co/{REPO_ID}")
    print("[INFO] High-speed Rust parallel transfer (hf_transfer): ACTIVE")

    commit_info = api.upload_folder(
        folder_path=str(BUNDLE_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Release Cortex-1 Large (421M): model weights, tokenizer, config & model card",
        token=token
    )

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Cortex-1 Large is now LIVE on Hugging Face Hub!")
    print(f"Commit: {commit_info}")
    print(f"URL:    https://huggingface.co/{REPO_ID}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else None
    upload_release(token=t)
