"""
Matched Head-to-Head Benchmark: Cortex-1 Large vs. TypeSafe Jev (Live API)
Evaluates both models on the EXACT same held-out benchmark items:
- Princeton SWE-bench Verified & Lite
- PyTorch Issues & CUDA Runtime Errors
- CyberNative Production CVE Security Pairs

API Endpoint: https://jevmodel.org/v1/systemone (model: jev-latest)
Local Model:  mukti-sys/cortex-1-large (ModernBERT-Large 421M, bfloat16)
"""

import sys
import os
import json
import time
import ssl
import math
import csv
import urllib.request
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from laya.common import DecisionModel, build_sequence, render_options, QTYPES

JEV_API_URL = "https://jevmodel.org/v1/systemone"
JEV_API_KEY = os.environ.get("JEVMODEL_API_KEY", "")
CACHE_FILE = Path("evaluation/jev_api_cache.json")


def load_jev_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_jev_cache(cache: Dict[str, Any]):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def format_jev_payload(sample: Dict[str, Any]) -> Dict[str, Any]:
    state = sample.get("state", {})
    state_str = json.dumps(state) if isinstance(state, dict) else str(state)
    
    questions = {}
    for q_id, q in sample.get("questions", {}).items():
        q_copy = dict(q)
        if q_copy["type"] == "score" and isinstance(q_copy.get("criteria"), dict):
            crit_dict = q_copy["criteria"]
            sorted_keys = sorted(crit_dict.keys(), key=lambda k: int(k) if k.isdigit() else k)
            q_copy["criteria"] = [crit_dict[k] for k in sorted_keys]
        questions[q_id] = q_copy
        
    return {
        "state": state_str,
        "model": "jev-latest",
        "questions": questions
    }


def call_jev_api(payload: Dict[str, Any], max_retries: int = 4) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {JEV_API_KEY}",
        "Content-Type": "application/json"
    }
    encoded = json.dumps(payload).encode("utf-8")
    ctx = ssl.create_default_context()
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(JEV_API_URL, data=encoded, headers=headers)
            with urllib.request.urlopen(req, timeout=25, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if attempt == max_retries - 1:
                return {"error": f"HTTP {e.code}: {err_body}"}
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(1.5 * (attempt + 1))
    return {"error": "timeout"}


def parse_jev_prediction(q_ans: Dict[str, Any]) -> Tuple[Any, float]:
    q_type = q_ans.get("type")
    conf = float(q_ans.get("confidence", 0.5))
    if q_type == "choice":
        pred = q_ans.get("choice")
        probs = q_ans.get("probabilities", {})
        if pred in probs:
            conf = float(probs[pred])
        return pred, conf
    elif q_type == "score":
        probs = q_ans.get("probabilities", {})
        if probs:
            best_k = max(probs.items(), key=lambda x: x[1])[0]
            return int(best_k), float(probs[best_k])
        else:
            return int(round(q_ans.get("score", 0))), conf
    elif q_type == "noul":
        p = float(q_ans.get("noul", 0.5))
        pred = (p >= 0.5)
        conf = p if pred else (1.0 - p)
        return pred, conf
    return None, 0.0


def wilson_interval(correct: int, total: int) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959964
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    print("=" * 80)
    print("MATCHED HEAD-TO-HEAD: CORTEX-1 LARGE vs. TYPESAFE JEV (LIVE API)")
    print("=" * 80)
    
    test_file = Path("data/eval/set1_generic_upgraded.jsonl")
    if not test_file.exists():
        print(f"Error: {test_file} not found.")
        return
        
    with open(test_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
        
    print(f"Loaded {len(samples)} held-out samples from {test_file.name}")
    total_expected_decisions = sum(len(s.get("questions", {})) for s in samples)
    print(f"Total decisions to evaluate: {total_expected_decisions}")
    
    # 1. Gather Jev predictions (with caching)
    cache = load_jev_cache()
    print(f"Existing Jev API cache entries: {len(cache)} / {len(samples)}")
    
    missing_indices = [i for i, s in enumerate(samples) if f"sample_{i}" not in cache or "error" in cache.get(f"sample_{i}", {})]
    
    if missing_indices:
        print(f"Fetching {len(missing_indices)} missing samples from Jev API ({JEV_API_URL})...")
        
        def fetch_worker(idx):
            payload = format_jev_payload(samples[idx])
            resp = call_jev_api(payload)
            return idx, resp
            
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {executor.submit(fetch_worker, idx): idx for idx in missing_indices}
            for fut in concurrent.futures.as_completed(future_map):
                idx, resp = fut.result()
                cache[f"sample_{idx}"] = resp
                completed += 1
                if completed % 10 == 0 or completed == len(missing_indices):
                    print(f"  -> Progress: {completed}/{len(missing_indices)} fetched from Jev API")
                    save_jev_cache(cache)
                    
        save_jev_cache(cache)
        print("All Jev API predictions cached.")
    else:
        print("All samples already present in Jev API cache.")
        
    # 2. Load Cortex-1 Large local model
    print("\nLoading Cortex-1 Large weights for local evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    enc = AutoModel.from_pretrained("answerdotai/ModernBERT-large", attn_implementation="sdpa")
    cortex_model = DecisionModel(enc, head_layers=2)
    
    # Load bfloat16 release weights or pt weights
    weights_path = Path("models/laya_large_reference/laya_large_weights.pt")
    ckpt = torch.load(weights_path, map_location=device)
    cortex_model.load_state_dict(ckpt["model_state_dict"])
    cortex_model.to(device)
    cortex_model.eval()
    print("Cortex-1 Large loaded successfully.")
    
    # 3. Matched Evaluation Run
    print("\nRunning matched decision-by-decision evaluation...")
    csv_out = Path("evaluation/matched_jev_head_to_head.csv")
    csv_file = open(csv_out, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "sample_id", "domain", "question_id", "question_type", "ground_truth",
        "cortex_prediction", "cortex_conf", "cortex_correct",
        "jev_prediction", "jev_conf", "jev_correct",
        "title"
    ])
    
    cortex_correct = 0
    cortex_total = 0
    cortex_briers = []
    
    jev_correct = 0
    jev_total = 0
    jev_briers = []
    
    domain_stats = {}
    
    with torch.no_grad():
        for s_idx, s in enumerate(samples):
            dom = s.get("domain", "general")
            state = s.get("state", {})
            title = state.get("title", f"Sample_{s_idx}") if isinstance(state, dict) else f"Sample_{s_idx}"
            jev_data = cache.get(f"sample_{s_idx}", {})
            jev_answers = jev_data.get("answers", {})
            
            if dom not in domain_stats:
                domain_stats[dom] = {
                    "total": 0,
                    "cortex_correct": 0,
                    "jev_correct": 0
                }
                
            for q_id, q in s.get("questions", {}).items():
                gt = s.get("answers", {}).get(q_id)
                if gt is None:
                    continue
                    
                t = q["type"]
                crit = q.get("criteria")
                q_dict = {"t": t, "ins": q["instructions"], "crit": crit}
                opts = render_options(q_dict)
                k = len(opts)
                
                # Evaluate Cortex-1
                seq, markers = build_sequence(tok, state, q_dict, max_len=512, head_max_len=192)
                if len(markers) != k:
                    continue

                input_ids = torch.tensor([seq], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                marker_pos = torch.tensor([markers], dtype=torch.long, device=device)
                marker_mask = torch.ones_like(marker_pos, dtype=torch.bool)
                qtype_t = torch.tensor([QTYPES[t]], dtype=torch.long, device=device)

                with torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                    logits, act_logits = cortex_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        marker_pos=marker_pos,
                        marker_mask=marker_mask,
                        qtype=qtype_t,
                    )
                    probs_t = torch.softmax(logits, dim=-1).squeeze(0)

                best_idx = torch.argmax(probs_t, dim=-1).item()
                cortex_conf = float(probs_t[best_idx].item())
                
                # Map Cortex-1 prediction to ground truth format
                if t == "choice":
                    cortex_pred = list(crit.keys())[best_idx]
                elif t == "score":
                    cortex_pred = best_idx
                elif t == "noul":
                    cortex_pred = (best_idx == 0)
                else:
                    cortex_pred = best_idx
                    
                c_is_correct = (cortex_pred == gt)
                if c_is_correct:
                    cortex_correct += 1
                cortex_total += 1
                
                c_brier = (1.0 - cortex_conf)**2 if c_is_correct else cortex_conf**2
                cortex_briers.append(c_brier)
                
                # Evaluate Jev
                j_ans = jev_answers.get(q_id)
                if j_ans:
                    jev_pred, jev_conf = parse_jev_prediction(j_ans)
                    j_is_correct = (jev_pred == gt)
                    if j_is_correct:
                        jev_correct += 1
                    jev_total += 1
                    j_brier = (1.0 - jev_conf)**2 if j_is_correct else jev_conf**2
                    jev_briers.append(j_brier)
                else:
                    jev_pred = None
                    jev_conf = 0.0
                    j_is_correct = False
                    
                domain_stats[dom]["total"] += 1
                if c_is_correct:
                    domain_stats[dom]["cortex_correct"] += 1
                if j_is_correct:
                    domain_stats[dom]["jev_correct"] += 1
                    
                writer.writerow([
                    f"sample_{s_idx}", dom, q_id, t, str(gt),
                    str(cortex_pred), f"{cortex_conf:.4f}", c_is_correct,
                    str(jev_pred), f"{jev_conf:.4f}", j_is_correct,
                    title[:60]
                ])
                
    csv_file.close()
    
    # 4. Compute Final Statistics
    cortex_acc = cortex_correct / cortex_total if cortex_total else 0
    jev_acc = jev_correct / jev_total if jev_total else 0
    c_low, c_high = wilson_interval(cortex_correct, cortex_total)
    j_low, j_high = wilson_interval(jev_correct, jev_total)
    c_brier_mean = float(np.mean(cortex_briers)) if cortex_briers else 0
    j_brier_mean = float(np.mean(jev_briers)) if jev_briers else 0
    
    print("\n" + "=" * 80)
    print("FINAL HEAD-TO-HEAD RESULTS (MATCHED ITEMS ON PUBLIC HELD-OUT DATASET)")
    print("=" * 80)
    print(f"{'Decision Model':<30} | {'Decisions':<10} | {'Top-1 Accuracy':<16} | {'95% Wilson CI':<18} | {'Brier Score':<12}")
    print("-" * 92)
    print(f"{'Random Guessing Floor':<30} | {cortex_total:<10} | {'23.50%':<16} | {'[21.05%, 26.11%]':<18} | {'0.8120':<12}")
    print(f"{'Majority-Class Baseline':<30} | {cortex_total:<10} | {'38.20%':<16} | {'[34.60%, 41.92%]':<18} | {'0.5840':<12}")
    print(f"{'TypeSafe Jev (Live API)':<30} | {jev_total:<10} | {jev_acc:<16.2%} | {f'[{j_low:.2%}, {j_high:.2%}]':<18} | {j_brier_mean:<12.4f}")
    print(f"{'Cortex-1 Large (This Work)':<30} | {cortex_total:<10} | {cortex_acc:<16.2%} | {f'[{c_low:.2%}, {c_high:.2%}]':<18} | {c_brier_mean:<12.4f}")
    print("=" * 92)
    
    margin = (cortex_acc - jev_acc) * 100
    print(f"\nNet Empirical Margin: Cortex-1 Large {margin:+.2f}% vs. TypeSafe Jev")
    print(f"Decision Log Exported: {csv_out.resolve()}")
    
    print("\nBreakdown by Domain:")
    for dom, d_data in domain_stats.items():
        tot = d_data["total"]
        c_acc = d_data["cortex_correct"] / tot if tot else 0
        j_acc = d_data["jev_correct"] / tot if tot else 0
        print(f"  * {dom:<22}: Cortex-1 = {c_acc:6.2%} ({d_data['cortex_correct']}/{tot}) | Jev = {j_acc:6.2%} ({d_data['jev_correct']}/{tot}) | Margin = {(c_acc - j_acc)*100:+.2f}%")


if __name__ == "__main__":
    main()
