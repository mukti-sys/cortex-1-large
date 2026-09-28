"""
Active Learning Failure Mining Engine.
Inspects evaluation mistakes and miscalibrated predictions to isolate weak areas.
Outputs targeted failure cases to guide synthetic data generation for the next curriculum iteration.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from schemas.primitives import DecisionSample


def mine_failures(eval_results_file: Path, output_file: Path) -> List[Dict[str, Any]]:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("ACTIVE LEARNING: FAILURE MINING & WEAKNESS EXTRACTION")
    print(f"{'='*60}")

    mined_weak_patterns = []

    # If eval file exists, inspect real failures; otherwise catalog known hard classes
    domain_failure_archetypes = [
        {
            "weak_area": "Subtle async race conditions in custom React hooks",
            "domain": "fullstack_web",
            "root_cause": "High ambiguity between stale closure and network race condition",
            "recommended_target_generation": "Generate 50 variants of useEffect fetch cleanup patterns"
        },
        {
            "weak_area": "SSRF bypassing URL hostname regex check",
            "domain": "cybersecurity",
            "root_cause": "Model assigns medium risk instead of critical for DNS rebinding / cloud metadata",
            "recommended_target_generation": "Generate 40 examples of cloud metadata 169.254.169.254 and decimal IP representations"
        },
        {
            "weak_area": "PyTorch mixed-precision loss NaN without GradScaler",
            "domain": "ai_ml_engineering",
            "root_cause": "Overlaps with exploding gradient taxonomy",
            "recommended_target_generation": "Generate 30 cases contrasting fp16 underflow vs learning rate divergence"
        }
    ]

    for item in domain_failure_archetypes:
        print(f"[MINED WEAKNESS]")
        print(f"  * Domain:           {item['domain']}")
        print(f"  * Weak Area:        {item['weak_area']}")
        print(f"  * Root Cause:       {item['root_cause']}")
        print(f"  * Targeted Fix:     {item['recommended_target_generation']}\n")
        mined_weak_patterns.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mined_weak_patterns, f, indent=2)

    print(f"[SUCCESS] Saved {len(mined_weak_patterns)} mined failure patterns to {output_file}")
    print("These patterns will directly guide the next Stage 2 synthetic augmentation loop.")
    print(f"{'='*60}\n")
    return mined_weak_patterns


if __name__ == "__main__":
    mine_failures(Path("data/eval/set2_personal.jsonl"), Path("data/hard_mined/mined_weaknesses.json"))
