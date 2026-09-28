"""
Comprehensive Benchmark Evaluation Engine for Laya ModernBERT-large Reference Architecture.
Evaluates frozen checkpoint across:
1. data/stage2_val.jsonl (Quarantined Validation Split & Keyword-Free Check)
2. data/eval/set2_personal.jsonl (Personal Held-Out Benchmark)
3. data/eval/set1_generic.jsonl (Generic Technical Benchmark vs Jev)
Uses dynamic marker-pooling (torch.gather at option [MASK] tokens) with zero static slot shortcuts.
Reports Sensitivity, Specificity-on-False, Balanced Accuracy, and Confusion Matrices.
"""

import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer, AutoModel
from laya.common import DecisionModel, build_sequence, render_options, QTYPES, QTYPE_NAMES


def format_state(st) -> str:
    if hasattr(st, "model_dump"):
        st = st.model_dump()
    if isinstance(st, dict):
        return f"Title: {st.get('title', '')}\nFile: {st.get('file_path', '')}\nContext: {st.get('context', '')}\nCode:\n{st.get('code_snippet', '')}"
    return str(st)


def evaluate_dataset(model, tok, jsonl_path: Path, benchmark_name: str, device) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {benchmark_name}")
    print(f"Dataset:   {jsonl_path.name}")
    print(f"{'='*70}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    total_decisions = 0
    correct_decisions = 0
    brier_scores = []

    domain_stats = {}
    question_stats = {}
    category_stats = {
        "choice": {"total": 0, "correct": 0},
        "score": {"total": 0, "correct": 0, "adjacent": 0},
        "noul": {
            "total": 0, "correct": 0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "subtle_total": 0, "subtle_tn": 0
        }
    }

    model.eval()

    with torch.no_grad():
        for s in samples:
            dom = s.get("domain", "general")
            if dom not in domain_stats:
                domain_stats[dom] = {"total": 0, "correct": 0}

            state = s.get("state", {})
            is_subtle = state.get("metadata", {}).get("is_keyword_free", False) if isinstance(state, dict) else False

            for q_id, q in s["questions"].items():
                ans = s["answers"].get(q_id)
                if ans is None:
                    continue

                t = q["type"]
                crit = q.get("criteria")
                q_dict = {"t": t, "ins": q["instructions"], "crit": crit}
                opts = render_options(q_dict)
                k = len(opts)

                seq, markers = build_sequence(tok, state, q_dict, max_len=512, head_max_len=192)
                if len(markers) != k:
                    continue

                input_ids = torch.tensor([seq], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                marker_pos = torch.tensor([markers], dtype=torch.long, device=device)
                marker_mask = torch.ones_like(marker_pos, dtype=torch.bool)
                qtype_t = torch.tensor([QTYPES[t]], dtype=torch.long, device=device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype_t)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)

                pred_idx = torch.argmax(probs, dim=-1).item()

                if q_id not in question_stats:
                    question_stats[q_id] = {"type": t, "total": 0, "correct": 0, "adjacent": 0}

                total_decisions += 1
                domain_stats[dom]["total"] += 1
                category_stats[t]["total"] += 1
                question_stats[q_id]["total"] += 1

                is_correct = False

                if t == "noul":
                    # Option 0 = False, Option 1 = True
                    target_label = 1 if ans is True else 0
                    is_correct = (pred_idx == target_label)
                    prob_true = probs[1].item()
                    target_num = 1.0 if ans is True else 0.0
                    brier_scores.append((prob_true - target_num) ** 2)

                    if target_label == 1:
                        if pred_idx == 1:
                            category_stats["noul"]["tp"] += 1
                        else:
                            category_stats["noul"]["fn"] += 1
                    else:
                        if is_subtle:
                            category_stats["noul"]["subtle_total"] += 1
                        if pred_idx == 0:
                            category_stats["noul"]["tn"] += 1
                            if is_subtle:
                                category_stats["noul"]["subtle_tn"] += 1
                        else:
                            category_stats["noul"]["fp"] += 1

                elif t == "score":
                    target_score = int(ans)
                    is_correct = (pred_idx == target_score)
                    if abs(pred_idx - target_score) <= 1:
                        category_stats["score"]["adjacent"] += 1
                        question_stats[q_id]["adjacent"] += 1

                elif t == "choice":
                    keys = list(crit.keys()) if isinstance(crit, dict) else []
                    target_idx = keys.index(ans) if ans in keys else 0
                    is_correct = (pred_idx == target_idx)

                if is_correct:
                    correct_decisions += 1
                    domain_stats[dom]["correct"] += 1
                    category_stats[t]["correct"] += 1
                    question_stats[q_id]["correct"] += 1

    overall_acc = correct_decisions / max(1, total_decisions)
    avg_brier = np.mean(brier_scores) if brier_scores else 0.0

    print(f"Total Decisions Evaluated: {total_decisions}")
    print(f"Real Overall Top-1 Accuracy: {overall_acc*100:.2f}% ({correct_decisions}/{total_decisions})")
    print(f"Mean Brier Score:           {avg_brier:.4f}")
    print("-" * 70)

    # Primitive Reporting
    print("Breakdown by Decision Primitive:")
    for cat, cdata in category_stats.items():
        if cdata["total"] > 0:
            c_acc = cdata["correct"] / cdata["total"]
            adj_str = f" [Adjacent +-1: {cdata['adjacent']}/{cdata['total']} ({cdata['adjacent']/cdata['total']*100:.1f}%)]" if cat == "score" else ""
            print(f"  * {cat.upper()}: {cdata['correct']}/{cdata['total']} ({c_acc*100:.1f}%){adj_str}")

    if category_stats["noul"]["total"] > 0:
        ns = category_stats["noul"]
        tp, tn, fp, fn = ns["tp"], ns["tn"], ns["fp"], ns["fn"]
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        bal_acc = (sens + spec) / 2.0
        subtle_spec = ns["subtle_tn"] / max(1, ns["subtle_total"])

        print("-" * 70)
        print("Autopilot Gating (NOUL) Tri-Metrics:")
        print(f"  * Sensitivity (Recall on True):  {sens*100:.1f}% ({tp}/{tp+fn})")
        print(f"  * Specificity (Recall on False): {spec*100:.1f}% ({tn}/{tn+fp})  <-- CRITICAL STOP-RATE")
        print(f"  * Balanced Accuracy:            {bal_acc*100:.1f}%")
        if ns["subtle_total"] > 0:
            print(f"  * Keyword-Free Subtle Recall:   {subtle_spec*100:.1f}% ({ns['subtle_tn']}/{ns['subtle_total']})")
        print(f"  * Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    print("-" * 70)
    print("Breakdown by Specific Question:")
    for qname, qdata in question_stats.items():
        q_acc = qdata["correct"] / max(1, qdata["total"])
        adj_str = f" [Adj: {qdata['adjacent']}/{qdata['total']} ({qdata['adjacent']/qdata['total']*100:.1f}%)]" if qdata["type"] == "score" else ""
        print(f"  * {qname} ({qdata['type']}): {qdata['correct']}/{qdata['total']} ({q_acc*100:.1f}%){adj_str}")

    print("-" * 70)
    print("Breakdown by Domain:")
    for dom, ddata in domain_stats.items():
        if ddata["total"] > 0:
            d_acc = ddata["correct"] / ddata["total"]
            print(f"  * {dom}: {ddata['correct']}/{ddata['total']} ({d_acc*100:.1f}%)")
    print(f"{'='*70}\n")

    return {
        "benchmark": benchmark_name,
        "total": total_decisions,
        "accuracy": overall_acc,
        "brier": avg_brier,
        "category_stats": category_stats,
        "question_stats": question_stats,
        "domain_stats": domain_stats
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating with ModernBERT-large reference architecture on {device}...")

    ckpt_path = Path("models/laya_large_reference/laya_large_weights.pt")
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    enc = AutoModel.from_pretrained("answerdotai/ModernBERT-large", attn_implementation="sdpa")
    model = DecisionModel(enc, head_layers=2)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 1. Quarantined Real Industry Validation Set
    p_val_upgraded = Path("data/upgraded_val.jsonl")
    p_val_legacy = Path("data/stage2_val.jsonl")
    if p_val_upgraded.exists():
        evaluate_dataset(model, tok, p_val_upgraded, "Real Industry Quarantined Validation (190 Samples / 570 Decisions)", device)
    elif p_val_legacy.exists():
        evaluate_dataset(model, tok, p_val_legacy, "Stage 2 Quarantined Validation (200 Samples)", device)

    # 2. Production Autopilot Gating Benchmark (Frozen Test)
    p_set2_upgraded = Path("data/eval/set2_personal_upgraded.jsonl")
    p_set2_legacy = Path("data/eval/set2_personal.jsonl")
    if p_set2_upgraded.exists():
        evaluate_dataset(model, tok, p_set2_upgraded, "Set 2: Real Production Gating Benchmark (60 Samples / 180 Decisions)", device)
    elif p_set2_legacy.exists():
        evaluate_dataset(model, tok, p_set2_legacy, "Set 2: Personal Held-Out Benchmark (104 Samples / 312 Decisions)", device)

    # 3. Real Generic SWE & Security Benchmark vs Jev (Frozen Test)
    p_set1_upgraded = Path("data/eval/set1_generic_upgraded.jsonl")
    p_set1_legacy = Path("data/eval/set1_generic.jsonl")
    if p_set1_upgraded.exists():
        evaluate_dataset(model, tok, p_set1_upgraded, "Set 1: Real Industry SWE & Security Benchmark vs Jev (130 Samples / 390 Decisions)", device)
    elif p_set1_legacy.exists():
        evaluate_dataset(model, tok, p_set1_legacy, "Set 1: Generic Technical Benchmark vs Jev (100 Samples / 300 Decisions)", device)


if __name__ == "__main__":
    main()
