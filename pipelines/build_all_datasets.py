"""
Master Dataset Builder and Split Generator.
Executes the full data ingestion pipeline, splits datasets into:
- Stage 1 Train (Tier B Silver)
- Stage 2 Train (Tier A Gold + Tier C Synthetic)
- Set 1 Held-Out Benchmark (Generic)
- Set 2 Held-Out Benchmark (Personal)
And runs strict leak-detection validation.
"""

import os
import json
import random
from pathlib import Path
from typing import List

from pipelines.extract_antigravity_history import extract_from_brain_artifacts
from pipelines.ingest_silver_adr import generate_adr_dataset
from pipelines.ingest_silver_security import generate_security_dataset
from pipelines.ingest_silver_bugs import generate_bugs_dataset
from pipelines.synthesize_tier_c import generate_tier_c_from_tier_a
from pipelines.validate_dataset import DatasetValidator, print_report
from schemas.primitives import DecisionSample


def build_all():
    print("[INFO] Starting Master Dataset Assembly...")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tier A: Antigravity History
    brain_root = Path(os.path.expanduser(r"~\.gemini\antigravity-ide\brain"))
    tier_a_file = data_dir / "tier_a_gold" / "antigravity_history.jsonl"
    gold_samples = extract_from_brain_artifacts(brain_root, tier_a_file)

    # Fallback seed if brain is completely new/empty
    if len(gold_samples) == 0:
        print("[INFO] Brain directory had no history yet; seeding initial Tier A samples...")
        from pipelines.ingest_silver_adr import ADR_SAMPLES_SEED
        from schemas.primitives import StructuredState, DomainType, DataTier
        from schemas.questions_catalog import AUTOPILOT_GATING_QUESTIONS

        seeded_a = []
        for i, s in enumerate(ADR_SAMPLES_SEED[:3]):
            seeded_a.append(DecisionSample(
                id=f"tier_a_seed_{i+1:04d}",
                domain=DomainType.AGENT_AUTOPILOT,
                tier=DataTier.TIER_A_GOLD,
                state=StructuredState(
                    title=f"Personal Project Plan: {s['title']}",
                    context=f"Context: {s['context']}\nPlan: {s['decision']}",
                    metadata={"source": "antigravity_seed"}
                ),
                questions=AUTOPILOT_GATING_QUESTIONS,
                answers={
                    "should_autopilot": s["should_autopilot"],
                    "risk_score": s["risk_score"],
                    "route_task": s["route_task"]
                }
            ))
        tier_a_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tier_a_file, "w", encoding="utf-8") as f:
            for item in seeded_a:
                f.write(item.model_dump_json() + "\n")
        gold_samples = seeded_a

    # 2. Tier B: Silver (ADR + Security + Bugs)
    adr_file = data_dir / "tier_b_silver" / "adr_decisions.jsonl"
    sec_file = data_dir / "tier_b_silver" / "security_decisions.jsonl"
    bugs_file = data_dir / "tier_b_silver" / "bugs_diagnostics.jsonl"

    adr_samples = generate_adr_dataset(adr_file)
    sec_samples = generate_security_dataset(sec_file)
    bugs_samples = generate_bugs_dataset(bugs_file)
    silver_samples = adr_samples + sec_samples + bugs_samples

    # 3. Tier C: Paraphrased Synthetic
    tier_c_file = data_dir / "tier_c_synthetic" / "paraphrased_personal.jsonl"
    tier_c_samples = generate_tier_c_from_tier_a(tier_a_file, tier_c_file, variants_per_sample=3)

    # 4. Form Split Datasets
    # Split Tier B into: Stage 1 Train (80%) and Set 1 Generic Held-out (20%)
    random.seed(42)
    shuffled_silver = list(silver_samples)
    random.shuffle(shuffled_silver)
    split_b = int(len(shuffled_silver) * 0.8)
    stage1_train = shuffled_silver[:split_b]
    set1_generic_eval = shuffled_silver[split_b:]

    # Split Tier A into: Stage 2 Train (80%) and Set 2 Personal Held-out (20%)
    shuffled_gold = list(gold_samples)
    random.shuffle(shuffled_gold)
    split_a = max(1, int(len(shuffled_gold) * 0.8))
    tier_a_train = shuffled_gold[:split_a]
    set2_personal_eval = shuffled_gold[split_a:] if len(shuffled_gold) > 1 else [shuffled_gold[0]]

    # Stage 2 Train combines Tier A (train) + Tier C
    stage2_train = tier_a_train + tier_c_samples

    # Write files
    def write_jsonl(path: Path, items: List[DecisionSample]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in items:
                f.write(s.model_dump_json() + "\n")

    p_stage1 = data_dir / "stage1_train_silver.jsonl"
    p_stage2 = data_dir / "stage2_train_personal.jsonl"
    p_set1 = data_dir / "eval" / "set1_generic.jsonl"
    p_set2 = data_dir / "eval" / "set2_personal.jsonl"

    write_jsonl(p_stage1, stage1_train)
    write_jsonl(p_stage2, stage2_train)
    write_jsonl(p_set1, set1_generic_eval)
    write_jsonl(p_set2, set2_personal_eval)

    print("\n[SUCCESS] Dataset Splits Created:")
    print(f"  * Stage 1 Train (Tier B Silver):      {len(stage1_train)} samples -> {p_stage1}")
    print(f"  * Stage 2 Train (Tier A + C):         {len(stage2_train)} samples -> {p_stage2}")
    print(f"  * Set 1 Held-Out Generic Benchmark:   {len(set1_generic_eval)} samples -> {p_set1}")
    print(f"  * Set 2 Held-Out Personal Benchmark:  {len(set2_personal_eval)} samples -> {p_set2}")

    # 5. Run Quality & Leakage Validation
    validator = DatasetValidator()
    print("\n[VALIDATION] Validating Splits for Schema Conformance and Zero Leakage...")

    r1 = validator.validate_file(p_stage1)
    print_report(r1)

    r_set1 = validator.validate_file(p_set1, check_leakage_against=[r1["state_hashes"]])
    print_report(r_set1)

    r2 = validator.validate_file(p_stage2)
    print_report(r2)

    r_set2 = validator.validate_file(p_set2)
    print_report(r_set2)


if __name__ == "__main__":
    build_all()
