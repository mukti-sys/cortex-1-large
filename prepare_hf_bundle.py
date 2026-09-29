"""
Prepares the complete, professional, standalone release bundle for Hugging Face Hub.
Packages:
- model.safetensors (1.68 GB weights)
- config.json (architecture hyperparameters)
- calibration_temperatures.json (temperature scaling)
- README.md (full model card with YAML frontmatter)
- assets/ (logo and banner)
- Complete ModernBERT tokenizer artifacts (tokenizer.json, vocab, config)
"""

import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "hf_release_bundle"

def create_bundle():
    print(f"[INFO] Creating clean unified release bundle at: {BUNDLE}")
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True, exist_ok=True)

    # 1. Weights
    src_weights = ROOT / "models" / "laya_large_reference" / "model.safetensors"
    dst_weights = BUNDLE / "model.safetensors"
    print(f"[1/6] Staging model.safetensors ({src_weights.stat().st_size / (1024*1024):.2f} MB)...")
    # Use hardlink or copy to save disk space if same volume
    try:
        os.link(src_weights, dst_weights)
        print("  * Linked via hardlink.")
    except Exception:
        shutil.copy2(src_weights, dst_weights)
        print("  * Copied file.")

    # 2. Config
    src_cfg = ROOT / "models" / "laya_large_reference" / "rl_agent_config.json"
    print("[2/6] Staging config.json...")
    shutil.copy2(src_cfg, BUNDLE / "config.json")

    # 3. Calibration
    src_calib = ROOT / "models" / "calibration" / "calibration_temperatures.json"
    print("[3/6] Staging calibration_temperatures.json...")
    shutil.copy2(src_calib, BUNDLE / "calibration_temperatures.json")

    # 4. Model Card
    src_card = ROOT / "model_card.md"
    print("[4/6] Staging README.md...")
    shutil.copy2(src_card, BUNDLE / "README.md")

    # 5. Assets
    assets_dir = BUNDLE / "assets"
    assets_dir.mkdir(exist_ok=True)
    print("[5/6] Staging assets/...")
    shutil.copy2(ROOT / "assets" / "cortex_logo.jpg", assets_dir / "cortex_logo.jpg")
    shutil.copy2(ROOT / "assets" / "cortex_banner.jpg", assets_dir / "cortex_banner.jpg")

    # 6. Standalone Tokenizer Artifacts
    print("[6/6] Packaging ModernBERT-large tokenizer for direct AutoTokenizer loading...")
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    tok.save_pretrained(BUNDLE)

    print("\n[SUCCESS] Unified release bundle prepared successfully!")
    for item in sorted(BUNDLE.rglob("*")):
        if item.is_file():
            rel = item.relative_to(BUNDLE)
            size = item.stat().st_size / (1024 * 1024)
            print(f"  - {rel} ({size:.2f} MB)")

if __name__ == "__main__":
    import os
    create_bundle()
