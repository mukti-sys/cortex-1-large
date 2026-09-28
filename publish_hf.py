"""
Hugging Face Hub Publisher for Cortex-1 Large (421M).
Uploads model weights (safetensors), model card, configuration, and calibration parameters
to https://huggingface.co/mukti-sys/cortex-1-large.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "mukti-sys/cortex-1-large"
REPO_TYPE = "model"


def publish(token: str = None):
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # 1. Verify Authentication
    try:
        user_info = api.whoami()
        print(f"[INFO] Authenticated as Hugging Face user: {user_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"[ERROR] Hugging Face authentication failed: {e}")
        print("\nPlease provide a valid HF Write Token via:")
        print("  1. Setting HF_TOKEN environment variable: $env:HF_TOKEN='hf_...'")
        print("  2. Running: huggingface-cli login")
        print("  3. Passing it directly to this script: python publish_hf.py --token hf_...")
        return False

    # 2. Create / Verify Model Repository
    print(f"[INFO] Creating or verifying model repository: {REPO_ID}...")
    try:
        repo_url = create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            exist_ok=True,
            token=token
        )
        print(f"[SUCCESS] Repository ready: {repo_url}")
    except Exception as e:
        print(f"[ERROR] Failed to create or access repository: {e}")
        return False

    # 3. Prepare Files for Upload
    root_dir = Path(__file__).resolve().parent
    files_to_upload = [
        # (local_path, repo_path)
        (root_dir / "model_card.md", "README.md"),
        (root_dir / "models" / "laya_large_reference" / "model.safetensors", "model.safetensors"),
        (root_dir / "models" / "laya_large_reference" / "rl_agent_config.json", "config.json"),
        (root_dir / "models" / "calibration" / "calibration_temperatures.json", "calibration_temperatures.json"),
        (root_dir / "assets" / "cortex_logo.jpg", "assets/cortex_logo.jpg"),
        (root_dir / "assets" / "cortex_banner.jpg", "assets/cortex_banner.jpg"),
    ]

    # Verify all files exist
    for local_p, target_name in files_to_upload:
        if not local_p.exists():
            print(f"[ERROR] Missing file required for upload: {local_p}")
            return False
        size_mb = local_p.stat().st_size / (1024 * 1024)
        print(f"  * Ready: {target_name} ({size_mb:.2f} MB)")

    # 4. Upload Files
    print("\n[INFO] Starting upload to Hugging Face Hub (this may take a few minutes for the 1.68 GB weights)...")
    for local_p, target_name in files_to_upload:
        print(f"[UPLOADING] {target_name}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=target_name,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token
            )
            print(f"[UPLOADED] {target_name} ✓")
        except Exception as e:
            print(f"[ERROR] Failed uploading {target_name}: {e}")
            return False

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Cortex-1 Large is now live on Hugging Face Hub!")
    print(f"URL: https://huggingface.co/{REPO_ID}")
    print("=" * 70)
    return True


if __name__ == "__main__":
    cli_token = None
    for arg in sys.argv[1:]:
        if arg.startswith("--token="):
            cli_token = arg.split("=", 1)[1]
        elif arg == "--token" and len(sys.argv) > sys.argv.index(arg) + 1:
            cli_token = sys.argv[sys.argv.index(arg) + 1]

    publish(token=cli_token)
