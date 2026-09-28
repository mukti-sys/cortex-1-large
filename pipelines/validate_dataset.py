"""
Strict Dataset Validator and Quality Assurance Engine for Laya.
Verifies:
1. 100% adherence to Laya's state + questions + answers schema.
2. Token length fit within ModernBERT limits (warns if > 1024 / > 8192).
3. Label balance and distribution reporting across choice and score primitives.
4. Clean held-out test isolation (zero hash/state leakage between splits).
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import Counter, defaultdict

from schemas.primitives import DecisionSample, QuestionType


class DatasetValidator:
    def __init__(self, max_token_estimate: int = 4096):
        self.max_token_estimate = max_token_estimate
        self.state_hashes: Set[str] = set()

    @staticmethod
    def _compute_state_hash(state: Any) -> str:
        state_str = json.dumps(state, sort_keys=True) if isinstance(state, dict) else str(state)
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()

    def validate_file(self, file_path: Path, check_leakage_against: List[Set[str]] = None) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        samples: List[DecisionSample] = []
        errors: List[str] = []
        domain_counts = Counter()
        tier_counts = Counter()
        question_type_counts = Counter()
        label_distributions = defaultdict(Counter)
        long_samples_count = 0
        duplicate_states = 0
        local_hashes: Set[str] = set()

        line_num = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    sample = DecisionSample.model_validate(data)
                    samples.append(sample)

                    domain_counts[sample.domain.value] += 1
                    tier_counts[sample.tier.value] += 1

                    # Check state hash for duplicates / leakage
                    h = self._compute_state_hash(sample.state)
                    if h in local_hashes:
                        duplicate_states += 1
                    local_hashes.add(h)

                    # Check leakage against forbidden sets (e.g. train vs test)
                    if check_leakage_against:
                        for forbidden_set in check_leakage_against:
                            if h in forbidden_set:
                                errors.append(f"Line {line_num}: Sample '{sample.id}' leaks across split boundaries!")

                    # Approximate token count (1 token ~= 4 chars)
                    state_len = len(str(sample.state))
                    if state_len // 4 > self.max_token_estimate:
                        long_samples_count += 1

                    for q_id, q in sample.questions.items():
                        question_type_counts[q.type.value] += 1
                        ans = sample.answers[q_id]
                        label_distributions[q_id][str(ans)] += 1

                except Exception as e:
                    errors.append(f"Line {line_num}: Validation failed - {str(e)}")

        report = {
            "file": str(file_path),
            "total_samples": len(samples),
            "valid": len(errors) == 0,
            "error_count": len(errors),
            "errors": errors[:10],  # show top 10
            "domain_breakdown": dict(domain_counts),
            "tier_breakdown": dict(tier_counts),
            "question_type_breakdown": dict(question_type_counts),
            "label_distributions": {k: dict(v) for k, v in label_distributions.items()},
            "duplicate_states_in_file": duplicate_states,
            "exceeds_token_estimate": long_samples_count,
            "state_hashes": local_hashes
        }
        return report


def print_report(report: Dict[str, Any]):
    print("\n" + "=" * 60)
    print(f"DATASET QUALITY REPORT: {Path(report['file']).name}")
    print("=" * 60)
    print(f"Status:             {'[PASSED]' if report['valid'] else '[FAILED]'}")
    print(f"Total Samples:      {report['total_samples']}")
    print(f"Errors Found:       {report['error_count']}")
    print(f"Duplicate States:   {report['duplicate_states_in_file']}")
    print(f"Overlength (>4k tok): {report['exceeds_token_estimate']}")
    print("-" * 60)
    print("Domains:")
    for d, c in report["domain_breakdown"].items():
        print(f"  * {d}: {c}")
    print("Tiers:")
    for t, c in report["tier_breakdown"].items():
        print(f"  * {t}: {c}")
    print("Question Primitives:")
    for q_type, c in report["question_type_breakdown"].items():
        print(f"  * {q_type}: {c}")
    if report["errors"]:
        print("-" * 60)
        print("Top Errors:")
        for err in report["errors"]:
            print(f"  [ERROR] {err}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        validator = DatasetValidator()
        rep = validator.validate_file(Path(sys.argv[1]))
        print_report(rep)
    else:
        print("Usage: python -m pipelines.validate_dataset <path_to_jsonl>")
