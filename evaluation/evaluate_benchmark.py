"""
Real Benchmark Evaluation Engine for Laya Decision Model.
Performs real tokenization, real ModernBERT forward pass, and compares
actual model predictions against ground truth labels on:
- Set 1: Generic Technical Benchmark (100 samples, vs Jev)
- Set 2: Personal Held-Out Benchmark (104 samples)
Computes real Top-1 Accuracy, real Brier Score, and per-domain accuracy breakdown.
"""

import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from transformers import AutoTokenizer
from training.train_laya import RealLayaDecisionModel
from schemas.primitives import DecisionSample, QuestionType


def run_real_evaluation(dataset_path: Path, checkpoint_path: Path, benchmark_name: str) -> Dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    print(f"\n{'='*60}")
    print(f"REAL BENCHMARK EVALUATION: {benchmark_name}")
    print(f"Dataset:            {dataset_path.name}")
    print(f"Model Checkpoint:   {checkpoint_path.name if checkpoint_path.exists() else 'Untrained Base'}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = RealLayaDecisionModel("answerdotai/ModernBERT-base")

    if checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"[INFO] Loaded trained weights from: {checkpoint_path}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint weights ({e}); evaluating with initialized weights.")

    model.to(device)
    model.eval()

    samples: List[DecisionSample] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(DecisionSample.model_validate(json.loads(line)))

    def format_state(st) -> str:
        if hasattr(st, "model_dump"):
            st = st.model_dump()
        if isinstance(st, dict):
            return f"Title: {st.get('title', '')}\nFile: {st.get('file_path', '')}\nContext: {st.get('context', '')}\nCode:\n{st.get('code_snippet', '')}"
        return str(st)

    total_decisions = 0
    correct_decisions = 0
    brier_scores = []
    domain_stats = {}
    question_stats = {}
    category_stats = {
        "choice": {"total": 0, "correct": 0},
        "score": {"total": 0, "correct": 0, "off_by_one": 0},
        "noul": {"total": 0, "correct": 0, "pred_true": 0, "pred_false": 0, "actual_true": 0, "actual_false": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    }

    with torch.no_grad():
        for s in samples:
            dom = s.domain.value
            if dom not in domain_stats:
                domain_stats[dom] = {"total": 0, "correct": 0}

            state_text = format_state(s.state)

            for q_id, q in s.questions.items():
                ans = s.answers.get(q_id)
                if ans is None:
                    continue

                if q_id not in question_stats:
                    question_stats[q_id] = {"type": q.type.value, "total": 0, "correct": 0, "off_by_one": 0}

                total_decisions += 1
                domain_stats[dom]["total"] += 1
                category_stats[q.type.value]["total"] += 1
                question_stats[q_id]["total"] += 1

                criteria_text = ""
                if q.criteria:
                    criteria_text = " | ".join([f"{k}: {v}" for k, v in q.criteria.items()])

                prompt = f"{state_text}\n[QUESTION]: {q.instructions}\n[CRITERIA]: {criteria_text}\n[DECISION]:"

                # Real tokenization and inference
                enc = tokenizer(prompt, max_length=256, truncation=True, padding=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                logits = model(input_ids, attention_mask, q.type.value)

                # Real decision evaluation
                is_correct = False
                if q.type == QuestionType.CHOICE:
                    keys = list(q.criteria.keys())
                    pred_idx = torch.argmax(logits, dim=-1).item()
                    pred_label = keys[pred_idx] if pred_idx < len(keys) else keys[0]
                    is_correct = (pred_label == ans)

                elif q.type == QuestionType.SCORE:
                    pred_score = torch.argmax(logits, dim=-1).item()
                    is_correct = (pred_score == int(ans))
                    if abs(pred_score - int(ans)) <= 1:
                        category_stats["score"]["off_by_one"] += 1
                        question_stats[q_id]["off_by_one"] += 1

                elif q.type == QuestionType.NOUL:
                    prob = torch.sigmoid(logits).item()
                    pred_bool = prob >= 0.5
                    target_num = 1.0 if ans is True else 0.0
                    brier_scores.append((prob - target_num) ** 2)
                    is_correct = (pred_bool == ans)

                    # Confusion tracking
                    if pred_bool:
                        category_stats["noul"]["pred_true"] += 1
                    else:
                        category_stats["noul"]["pred_false"] += 1

                    if ans is True:
                        category_stats["noul"]["actual_true"] += 1
                        if pred_bool:
                            category_stats["noul"]["tp"] += 1
                        else:
                            category_stats["noul"]["fn"] += 1
                    else:
                        category_stats["noul"]["actual_false"] += 1
                        if not pred_bool:
                            category_stats["noul"]["tn"] += 1
                        else:
                            category_stats["noul"]["fp"] += 1

                if is_correct:
                    correct_decisions += 1
                    domain_stats[dom]["correct"] += 1
                    category_stats[q.type.value]["correct"] += 1
                    question_stats[q_id]["correct"] += 1

    acc = correct_decisions / max(1, total_decisions)
    avg_brier = np.mean(brier_scores) if brier_scores else 0.0

    print(f"Total Decisions Evaluated: {total_decisions}")
    print(f"Real Overall Accuracy:    {acc:.4f} ({acc*100:.1f}%)")
    print(f"Real Mean Brier Score:    {avg_brier:.4f}")
    print("-" * 60)
    print("Breakdown by Decision Primitive:")
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            c_acc = stats["correct"] / stats["total"]
            extra = f" (Adjacent within 1: {stats['off_by_one']}/{stats['total']}, {stats['off_by_one']/stats['total']*100:.1f}%)" if cat == "score" else ""
            print(f"  * {cat.upper()}: {stats['correct']}/{stats['total']} ({c_acc*100:.1f}%){extra}")

    if category_stats["noul"]["total"] > 0:
        ns = category_stats["noul"]
        print("-" * 60)
        print("NOUL Binary Classification Diagnostics:")
        print(f"  * Actual Distribution:   True={ns['actual_true']}, False={ns['actual_false']}")
        print(f"  * Predicted Distribution: True={ns['pred_true']}, False={ns['pred_false']}")
        print(f"  * TP={ns['tp']}, FP={ns['fp']}, TN={ns['tn']}, FN={ns['fn']}")
        sens = ns['tp'] / max(1, ns['actual_true'])
        spec = ns['tn'] / max(1, ns['actual_false'])
        print(f"  * Sensitivity (Recall on True):  {sens*100:.1f}%")
        print(f"  * Specificity (Recall on False): {spec*100:.1f}%")

    print("-" * 60)
    print("Breakdown by Specific Question (Full Transparency):")
    for qname, qdata in question_stats.items():
        q_acc = qdata["correct"] / max(1, qdata["total"])
        extra = f" [Adjacent: {qdata['off_by_one']}/{qdata['total']}]" if qdata["type"] == "score" else ""
        print(f"  * {qname} ({qdata['type']}): {qdata['correct']}/{qdata['total']} ({q_acc*100:.1f}%){extra}")

    print("-" * 60)
    print("Breakdown by Technical Domain:")
    for dom, stats in domain_stats.items():
        if stats["total"] > 0:
            d_acc = stats["correct"] / stats["total"]
            print(f"  * {dom}: {stats['correct']}/{stats['total']} ({d_acc*100:.1f}%)")
    print(f"{'='*60}\n")

    return {
        "benchmark": benchmark_name,
        "total": total_decisions,
        "accuracy": acc,
        "brier_score": float(avg_brier),
        "domain_stats": domain_stats,
        "category_stats": category_stats
    }


if __name__ == "__main__":
    ref_ckpt = Path("models/laya_large_reference/laya_large_weights.pt")
    if ref_ckpt.exists():
        print("[INFO] Reference ModernBERT-large checkpoint detected. Delegating to reference evaluation engine...")
        from evaluation.evaluate_reference import main as ref_main
        ref_main()
    else:
        p_set1 = Path("data/eval/set1_generic.jsonl")
        p_set2 = Path("data/eval/set2_personal.jsonl")
        ckpt = Path("models/laya_final_brain/laya_real_weights.pt")

        if p_set1.exists():
            run_real_evaluation(p_set1, ckpt, "Set 1: Generic Technical Benchmark (100 Samples vs Jev)")
        if p_set2.exists():
            run_real_evaluation(p_set2, ckpt, "Set 2: Personal Held-Out Benchmark (104 Samples)")
