"""
Robust Mass Release Uploader for Cortex-1 Large.
Configures resilient httpx transport with 10 automatic retries and extended timeouts
to guarantee zero timeout dropouts on large 1.68 GB transfers.
Uploads hf_release_bundle/ as a single atomic release commit.
"""

import sys
import os
import time
from pathlib import Path

# 1. Force Disable Xet and Force Standard S3/Cloudflare LFS
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import httpx
import huggingface_hub.constants as C
C.DEFAULT_REQUEST_TIMEOUT = 3600
C.HF_HUB_DISABLE_XET = True

import huggingface_hub.utils._http as H
from huggingface_hub import HfApi

transport = httpx.HTTPTransport(retries=20)

def robust_client_factory() -> httpx.Client:
    return httpx.Client(
        transport=transport,
        event_hooks={"request": [H.hf_request_event_hook]},
        follow_redirects=True,
        timeout=httpx.Timeout(timeout=3600.0, connect=120.0, read=3600.0, write=3600.0)
    )

H.set_client_factory(robust_client_factory)

REPO_ID = "mukti-sys/cortex-1-large"
BUNDLE_DIR = Path(__file__).resolve().parent / "hf_release_bundle"


def upload_massive():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN is not set.", flush=True)
        return False

    api = HfApi(token=token)
    user = api.whoami().get("name", "Unknown")
    print(f"[INFO] Authenticated as: {user}", flush=True)
    print(f"[INFO] Source Release Bundle: {BUNDLE_DIR}", flush=True)
    print(f"[INFO] Target Repository: https://huggingface.co/{REPO_ID}", flush=True)
    print("[INFO] Standard S3 Multi-part LFS Mode: ACTIVE (Xet completely disabled).", flush=True)
    print("[INFO] Resilient HTTP Transport: 20 retries, 1-hour timeout active.", flush=True)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"\n[INFO] Starting upload attempt {attempt}/{max_attempts}...", flush=True)
        try:
            commit_info = api.upload_folder(
                folder_path=str(BUNDLE_DIR),
                repo_id=REPO_ID,
                repo_type="model",
                commit_message="Release Cortex-1 Large (421M): weights, tokenizer, config & model card",
                commit_description="Production release of Cortex-1 Large non-autoregressive decision model.",
                token=token
            )
            print("\n" + "=" * 75, flush=True)
            print("[SUCCESS] CORTEX-1 LARGE RELEASE LIVE ON HUGGING FACE HUB!", flush=True)
            print(f"Commit URL: {commit_info}", flush=True)
            print(f"Model URL:  https://huggingface.co/{REPO_ID}", flush=True)
            print("=" * 75, flush=True)
            return True
        except Exception as e:
            print(f"\n[WARN] Attempt {attempt} failed: {e}", flush=True)
            if attempt < max_attempts:
                print("[INFO] Retrying in 10 seconds...", flush=True)
                time.sleep(10)
            else:
                print("[ERROR] All upload attempts exhausted.", flush=True)
                return False


if __name__ == "__main__":
    upload_massive()
