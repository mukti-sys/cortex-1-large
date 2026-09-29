"""
Direct Head-to-Head Comparison: Base Laya vs. Cortex-1 Large on Real Coding & Security Decisions.
Evaluates on:
- Set 1: Real SWE-bench bugs, PyTorch ML errors, and CVE security vulnerabilities (690 decisions)
- Set 2: Real Developer PR Autopilot Gating (426 decisions)
"""

import sys
import json
import time
import math
import numpy as np
import torch
from pathlib import Path
import laya

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half_width = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))

def evaluate_file(agent, filepath: Path, model_name: str):
    total = 0
    correct = 0
    by_domain = {}
    gating_stats = {"safe_total": 0, "safe_approved": 0, "unsafe_total": 0, "unsafe_blocked": 0}
    brier_scores = []
    latencies = []

    with open(filepath, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    for s in samples:
        domain = s.get("domain", "general")
        state = s["state"]
        questions = s["questions"]
        answers_gold = s["answers"]

        norm_questions = {}
        for qid, qdef in questions.items():
            qd = dict(qdef)
            qtype = qd.get("type", "choice")
            crit = qd.get("criteria")
            if qtype == "score" and isinstance(crit, dict):
                try:
                    sorted_keys = sorted(crit.keys(), key=lambda x: int(x))
                    qd["criteria"] = [crit[k] for k in sorted_keys]
                except Exception:
                    qd["criteria"] = list(crit.values())
            norm_questions[qid] = qd

        t0 = time.perf_counter()
        pred = agent.predict(state, norm_questions)
        latencies.append((time.perf_counter() - t0) * 1000)

        ans_dict = pred.get("answers", {})

        for qid, qdef in questions.items():
            if qid not in answers_gold or qid not in ans_dict:
                continue

            gold_val = answers_gold[qid]
            ans = ans_dict[qid]
            qtype = qdef.get("type", "choice")

            if qtype == "choice":
                pred_label = ans.get("choice")
                gold_label = gold_val
                is_correct = (pred_label == gold_label)

                probs = ans.get("probabilities", {})
                crit = qdef.get("criteria", {})
                keys = list(crit.keys()) if isinstance(crit, dict) else []
                p_vec = [probs.get(k, 0.0) for k in keys]
                gold_vec = [1.0 if k == gold_label else 0.0 for k in keys]
                if p_vec and gold_vec:
                    brier_scores.append(sum((pv - gv)**2 for pv, gv in zip(p_vec, gold_vec)))

            elif qtype == "score":
                probs = ans.get("probabilities", {})
                pred_label = int(max(probs.items(), key=lambda x: x[1])[0]) if probs else round(ans.get("score", 0))
                gold_label = int(gold_val)
                is_correct = (pred_label == gold_label)

                crit = qdef.get("criteria", {})
                k = len(crit) if isinstance(crit, (dict, list)) else 5
                keys = [str(i) for i in range(k)]
                p_vec = [probs.get(k_i, 0.0) for k_i in keys]
                gold_vec = [1.0 if i == gold_label else 0.0 for i in range(k)]
                if p_vec and gold_vec:
                    brier_scores.append(sum((pv - gv)**2 for pv, gv in zip(p_vec, gold_vec)))

            elif qtype == "noul":
                p_true = ans.get("noul", 0.0)
                pred_label = (p_true >= 0.5)
                gold_label = bool(gold_val)
                is_correct = (pred_label == gold_label)
                p_vec = [1.0 - p_true, p_true]
                gold_vec = [0.0, 1.0] if gold_label else [1.0, 0.0]
                brier_scores.append(sum((pv - gv)**2 for pv, gv in zip(p_vec, gold_vec)))

                # Track gating specifically
                if qid in ("should_autopilot", "is_immediate_blocker", "requires_code_refactor"):
                    if gold_label is True:
                        gating_stats["safe_total"] += 1
                        if pred_label is True:
                            gating_stats["safe_approved"] += 1
                    else:
                        gating_stats["unsafe_total"] += 1
                        if pred_label is False:
                            gating_stats["unsafe_blocked"] += 1

            total += 1
            if is_correct:
                correct += 1

            if domain not in by_domain:
                by_domain[domain] = {"correct": 0, "total": 0}
            by_domain[domain]["total"] += 1
            if is_correct:
                by_domain[domain]["correct"] += 1

    acc = (correct / total) * 100 if total > 0 else 0
    ci_low, ci_high = wilson_interval(correct, total)
    mean_brier = float(np.mean(brier_scores)) if brier_scores else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    return {
        "model": model_name,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "ci": (ci_low, ci_high),
        "brier": mean_brier,
        "latency_ms": mean_lat,
        "by_domain": by_domain,
        "gating": gating_stats
    }

def main():
    print("=" * 80)
    print(" HEAD-TO-HEAD CODING & SECURITY BENCHMARK: BASE LAYA vs CORTEX-1 LARGE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    print("\nLoading models...")
    base_agent = laya.load("models/base_laya_typed_decisions", device=device)
    cortex_agent = laya.load("models/laya_large_reference", device=device)

    set1_path = Path("data/eval/set1_generic_upgraded.jsonl")
    set2_path = Path("data/eval/set2_personal_upgraded.jsonl")

    # 1. Evaluate Set 1 (SWE-bench, PyTorch ML, CVE Security - 690 decisions)
    print("\n[1/2] Evaluating Set 1: SWE-bench, PyTorch CUDA errors, CVE vulnerabilities (690 Decisions)...")
    res1_base = evaluate_file(base_agent, set1_path, "Base Laya")
    res1_cortex = evaluate_file(cortex_agent, set1_path, "Cortex-1 Large")

    # 2. Evaluate Set 2 (Developer PR Autopilot Gating - 426 decisions)
    print("\n[2/2] Evaluating Set 2: Developer PR Autopilot Gating (426 Decisions)...")
    res2_base = evaluate_file(base_agent, set2_path, "Base Laya")
    res2_cortex = evaluate_file(cortex_agent, set2_path, "Cortex-1 Large")

    # Print Report
    print("\n" + "=" * 80)
    print(" BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print("\n### SET 1: CODING, AI/ML DIAGNOSTICS & CYBERSECURITY (690 Decisions)")
    print("-" * 80)
    print(f"{'Model':<25} | {'Decisions':<9} | {'Accuracy':<10} | {'95% CI':<18} | {'Brier':<8} | {'Mean Latency':<12}")
    print("-" * 80)
    for r in [res1_base, res1_cortex]:
        ci_str = f"[{r['ci'][0]*100:.1f}%, {r['ci'][1]*100:.1f}%]"
        print(f"{r['model']:<25} | {r['total']:<9} | {r['accuracy']:>8.2f}% | {ci_str:<18} | {r['brier']:>6.4f} | {r['latency_ms']:>8.2f} ms")

    print("\n--- Domain Breakdown (Set 1) ---")
    domains = list(res1_cortex["by_domain"].keys())
    for d in domains:
        print(f"\n[Domain: {d}]")
        b_c = res1_base["by_domain"][d]["correct"]
        b_t = res1_base["by_domain"][d]["total"]
        c_c = res1_cortex["by_domain"][d]["correct"]
        c_t = res1_cortex["by_domain"][d]["total"]
        print(f"  - Base Laya    : {b_c}/{b_t} ({(b_c/b_t)*100:.1f}%)")
        print(f"  - Cortex-1     : {c_c}/{c_t} ({(c_c/c_t)*100:.1f}%) [Margin: +{((c_c/c_t) - (b_c/b_t))*100:.1f}%]")

    print("\n" + "=" * 80)
    print("### SET 2: DEVELOPER PR AUTOPILOT GATING (426 Decisions)")
    print("-" * 80)
    print(f"{'Model':<25} | {'Decisions':<9} | {'Accuracy':<10} | {'95% CI':<18} | {'Brier':<8}")
    print("-" * 80)
    for r in [res2_base, res2_cortex]:
        ci_str = f"[{r['ci'][0]*100:.1f}%, {r['ci'][1]*100:.1f}%]"
        print(f"{r['model']:<25} | {r['total']:<9} | {r['accuracy']:>8.2f}% | {ci_str:<18} | {r['brier']:>6.4f}")

    print("\n--- Autopilot Gating Quality (Sensitivity vs Specificity) ---")
    for r in [res2_base, res2_cortex]:
        g = r["gating"]
        sens = (g["safe_approved"] / g["safe_total"] * 100) if g["safe_total"] > 0 else 0
        spec = (g["unsafe_blocked"] / g["unsafe_total"] * 100) if g["unsafe_total"] > 0 else 0
        print(f"{r['model']}:")
        print(f"  * Safe PRs Approved (Sensitivity)    : {g['safe_approved']}/{g['safe_total']} ({sens:.1f}%)")
        print(f"  * Dangerous PRs Blocked (Specificity): {g['unsafe_blocked']}/{g['unsafe_total']} ({spec:.1f}%)")
        print(f"  * False Approvals (Safety Risk)      : {g['unsafe_total'] - g['unsafe_blocked']}/{g['unsafe_total']}")

if __name__ == "__main__":
    main()
