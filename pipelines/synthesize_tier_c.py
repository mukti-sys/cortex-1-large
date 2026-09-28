"""
Tier C (Synthetic) Paraphrasing & Augmentation Engine.
Paraphrases Tier A (Gold) personal examples to multiply personal training signal
without shifting ground truth labels or introducing synthetic bias.
"""

import json
import random
from pathlib import Path
from typing import List

from schemas.primitives import DecisionSample, StructuredState, DataTier


PARAPHRASE_PREFIXES = [
    "Refactored task proposal: ",
    "Engineering ticket: ",
    "Antigravity plan draft: ",
    "Architectural recommendation: ",
    "Developer prompt for review: "
]

PARAPHRASE_POSTFIXES = [
    " Please evaluate safety and risk profile before execution.",
    " Review against team security and breaking-change conventions.",
    " Check if this modification should run autonomously.",
    " Validate whether this task requires human review."
]


def paraphrase_state(original_state: StructuredState, variant_idx: int) -> StructuredState:
    orig_context = original_state.context or ""
    prefix = PARAPHRASE_PREFIXES[variant_idx % len(PARAPHRASE_PREFIXES)]
    postfix = PARAPHRASE_POSTFIXES[variant_idx % len(PARAPHRASE_POSTFIXES)]

    new_context = f"{prefix}{orig_context.strip()}{postfix}"
    new_title = f"{original_state.title or 'Task'} (Variant {variant_idx+1})"

    meta = dict(original_state.metadata or {})
    meta["paraphrase_variant"] = variant_idx + 1
    meta["source_tier"] = "tier_a_gold"

    return StructuredState(
        title=new_title,
        context=new_context,
        file_path=original_state.file_path,
        language=original_state.language,
        code_snippet=original_state.code_snippet,
        diff=original_state.diff,
        stack_trace=original_state.stack_trace,
        metadata=meta
    )


def generate_tier_c_from_tier_a(tier_a_path: Path, tier_c_path: Path, variants_per_sample: int = 3) -> List[DecisionSample]:
    if not tier_a_path.exists():
        print(f"Tier A file not found at {tier_a_path}. Run extract_antigravity_history.py first.")
        return []

    tier_c_samples: List[DecisionSample] = []
    sample_idx = 0

    with open(tier_a_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            gold_sample = DecisionSample.model_validate(json.loads(line))
            
            # Create synthetic variants strictly preserving gold ground-truth answers
            for v in range(variants_per_sample):
                sample_idx += 1
                if isinstance(gold_sample.state, StructuredState):
                    syn_state = paraphrase_state(gold_sample.state, v)
                else:
                    syn_state = f"{PARAPHRASE_PREFIXES[v % len(PARAPHRASE_PREFIXES)]}{gold_sample.state}"

                syn_sample = DecisionSample(
                    id=f"tier_c_syn_{sample_idx:04d}_from_{gold_sample.id}",
                    domain=gold_sample.domain,
                    tier=DataTier.TIER_C_SYNTHETIC,
                    state=syn_state,
                    questions=gold_sample.questions,
                    answers=gold_sample.answers  # Exact label preservation
                )
                tier_c_samples.append(syn_sample)

    tier_c_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tier_c_path, "w", encoding="utf-8") as f:
        for s in tier_c_samples:
            f.write(s.model_dump_json() + "\n")

    print(f"Generated {len(tier_c_samples)} Tier C (Synthetic) samples saved to {tier_c_path}")
    return tier_c_samples


if __name__ == "__main__":
    tier_a = Path("data/tier_a_gold/antigravity_history.jsonl")
    tier_c = Path("data/tier_c_synthetic/paraphrased_personal.jsonl")
    generate_tier_c_from_tier_a(tier_a, tier_c)
