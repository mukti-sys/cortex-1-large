"""
Real Local Cortex-1 Decision Engine MCP Server for Google Antigravity & Cursor.
Runs real ModernBERT-large transformer inference with multi-task decision heads.
Measures real neural network forward-pass latency.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, AutoModel
from laya.common import DecisionModel, build_sequence, render_options, QTYPES


class CortexDecisionServer:
    def __init__(self, checkpoint_path: str = "models/laya_large_reference/laya_large_weights.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        print(f"[INFO] Initializing Real Cortex-1 Decision Server on {self.device} ({dev_name})...")
        print(f"[INFO] Loading ModernBERT-large (421M) reference backbone...")
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        enc = AutoModel.from_pretrained("answerdotai/ModernBERT-large", attn_implementation="sdpa")
        self.model = DecisionModel(enc, head_layers=2)

        # Resolve model weights (local safetensors/pt or auto-fetch from Hugging Face Hub)
        resolved_ckpt = None
        candidates = [
            Path(checkpoint_path) if checkpoint_path else None,
            Path("models/laya_large_reference/model.safetensors"),
            Path("models/laya_large_reference/laya_large_weights.pt"),
            Path("model.safetensors")
        ]
        for c in candidates:
            if c and c.exists():
                resolved_ckpt = c
                break

        if resolved_ckpt is None:
            print("[INFO] No local weights found. Fetching mukti-sys/cortex-1-large from Hugging Face Hub...")
            try:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(repo_id="mukti-sys/cortex-1-large", filename="model.safetensors")
                resolved_ckpt = Path(downloaded)
            except Exception as e:
                print(f"[WARN] Could not download from Hugging Face Hub: {e}")

        if resolved_ckpt and resolved_ckpt.exists():
            try:
                if resolved_ckpt.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    state_dict = load_file(str(resolved_ckpt))
                    self.model.load_state_dict(state_dict)
                    print(f"[INFO] Loaded trained weights from: {resolved_ckpt}")
                else:
                    ckpt = torch.load(resolved_ckpt, map_location=self.device)
                    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                    self.model.load_state_dict(sd)
                    print(f"[INFO] Loaded trained weights from: {resolved_ckpt}")
            except Exception as e:
                print(f"[WARN] Error loading checkpoint from {resolved_ckpt}: {e}")

        self.model.to(self.device)
        self.model.eval()

        # Criteria mappings
        self.security_classes = [
            "sql_injection", "xss", "ssrf", "authz_idor", "command_injection", "crypto_secret_leak", "none_secure"
        ]
        self.routes = ["fast_system1", "balanced_agent", "deep_reasoning"]
        self.bug_classes = [
            "hydration_mismatch", "async_race_condition", "null_undefined_access", "cuda_oom", "tensor_shape_mismatch", "gradient_nan_inf"
        ]

        # Warm up CUDA kernels and tokenizer for steady-state low latency
        if self.device.type == "cuda":
            print(f"[INFO] Warming up RTX 5050 CUDA kernels for sub-35ms inference...")
            with torch.no_grad():
                for warmup_t, warmup_crit in [
                    ("noul", None),
                    ("score", {str(i): f"Level {i}" for i in range(5)}),
                    ("choice", {"opt_a": "Option A", "opt_b": "Option B", "opt_c": "Option C"})
                ]:
                    q_dummy = {"t": warmup_t, "ins": "warmup check", "crit": warmup_crit}
                    seq, markers = build_sequence(self.tokenizer, "warmup state context with sufficient token length to warm up attention kernels.", q_dummy)
                    in_ids = torch.tensor([seq], dtype=torch.long, device=self.device)
                    m_pos = torch.tensor([markers], dtype=torch.long, device=self.device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _ = self.model(in_ids, torch.ones_like(in_ids), m_pos, torch.ones_like(m_pos, dtype=torch.bool), torch.tensor([QTYPES[warmup_t]], device=self.device))
                torch.cuda.synchronize()
            print(f"[INFO] CUDA warmup complete. Ready for real inference.")

    def _infer_decision(self, state: Any, question_dict: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        t = question_dict["t"]
        seq, markers = build_sequence(self.tokenizer, state, question_dict, max_len=512, head_max_len=192)

        input_ids = torch.tensor([seq], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        marker_pos = torch.tensor([markers], dtype=torch.long, device=self.device)
        marker_mask = torch.ones_like(marker_pos, dtype=torch.bool)
        qtype_t = torch.tensor([QTYPES[t]], dtype=torch.long, device=self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, act_logits = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_latency_ms = (time.perf_counter() - start) * 1000
        return probs, gpu_latency_ms

    def should_autopilot(self, plan_text: str) -> Dict[str, Any]:
        q_noul = {
            "t": "noul",
            "ins": "Should this task or plan run unsupervised in autopilot mode without stopping for user approval?",
            "crit": None
        }
        probs_noul, latency_ms = self._infer_decision(plan_text, q_noul)
        prob_true = probs_noul[1].item()
        allow_unsupervised = bool(prob_true >= 0.5)

        q_score = {
            "t": "score",
            "ins": "Rate the architectural, data loss, or system regression risk of this change on a 0-4 scale.",
            "crit": {str(i): f"Level {i}" for i in range(5)}
        }
        probs_score, _ = self._infer_decision(plan_text, q_score)
        risk_score = int(torch.argmax(probs_score).item())
        route = self.routes[min(len(self.routes) - 1, max(0, risk_score - 1))]

        return {
            "allow_unsupervised": allow_unsupervised,
            "confidence": round(prob_true if allow_unsupervised else 1.0 - prob_true, 4),
            "risk_score": risk_score,
            "route_tier": route,
            "real_inference_latency_ms": round(latency_ms, 2),
            "model": "Cortex-1-Large-421M"
        }

    def risk_score(self, code_diff_or_plan: str) -> Dict[str, Any]:
        q_score = {
            "t": "score",
            "ins": "Rate the architectural, data loss, or system regression risk of this change on a 0-4 scale.",
            "crit": {str(i): f"Level {i}" for i in range(5)}
        }
        probs, latency_ms = self._infer_decision(code_diff_or_plan, q_score)
        score = int(torch.argmax(probs).item())

        rubrics = {
            0: "Cosmetic / zero runtime risk",
            1: "Low: localized tweak",
            2: "Medium: shared utility or database query",
            3: "High: core schema or API breaking change",
            4: "Critical: irreversible data deletion or security risk"
        }

        return {
            "score": score,
            "rubric_level": rubrics.get(score, "Medium"),
            "is_critical": score >= 3,
            "confidence": round(probs[score].item(), 4),
            "real_inference_latency_ms": round(latency_ms, 2)
        }

    def triage_security(self, code_snippet: str, file_path: str = "") -> Dict[str, Any]:
        state = {"code_snippet": code_snippet, "file_path": file_path}
        q_cwe = {
            "t": "choice",
            "ins": "Identify CWE vulnerability class.",
            "crit": {c: c.replace("_", " ") for c in self.security_classes}
        }
        probs_cwe, latency_ms = self._infer_decision(state, q_cwe)
        pred_idx = torch.argmax(probs_cwe).item()
        cwe = self.security_classes[pred_idx]

        q_block = {
            "t": "noul",
            "ins": "Must this security flaw block deployment or code commit immediately?",
            "crit": None
        }
        probs_block, _ = self._infer_decision(state, q_block)
        is_blocker = bool(probs_block[1].item() >= 0.5)

        q_score = {
            "t": "score",
            "ins": "Rate vulnerability exploitability score from 0 to 4.",
            "crit": {str(i): f"Severity {i}" for i in range(5)}
        }
        probs_score, _ = self._infer_decision(state, q_score)
        exploitability_score = int(torch.argmax(probs_score).item())

        strategies = {
            "sql_injection": "Use parameterized queries or ORM prepared statements; never interpolate raw variables into SQL.",
            "xss": "Context-aware HTML encoding and sanitization, or framework-safe templating with DOMPurify.",
            "ssrf": "Strict domain/IP allowlisting, validate URLs before fetching, block RFC1918 private IP ranges.",
            "authz_idor": "Enforce server-side user tenancy checks: verify current_user.id owns target object ID on every request.",
            "command_injection": "Pass arguments as an array to subprocess without shell=True; avoid os.system().",
            "crypto_secret_leak": "Rotate leaked credentials immediately; load secrets strictly via environment variables / secret manager.",
            "none_secure": "Code pattern adheres to secure coding practices."
        }
        suggested_strategy = strategies.get(cwe, "Conduct security review and follow OWASP remediation guidelines.")

        return {
            "vulnerability_class": cwe,
            "exploitability_score": exploitability_score,
            "is_immediate_blocker": is_blocker,
            "suggested_fix_strategy": suggested_strategy,
            "real_inference_latency_ms": round(latency_ms, 2)
        }

    def diagnose_root_cause(self, trace_or_snippet: str) -> Dict[str, Any]:
        q_bug = {
            "t": "choice",
            "ins": "What is the primary bug root cause?",
            "crit": {b: b.replace("_", " ") for b in self.bug_classes}
        }
        probs, latency_ms = self._infer_decision(trace_or_snippet, q_bug)
        pred_idx = torch.argmax(probs).item()
        bug = self.bug_classes[pred_idx]

        fix_actions = {
            "hydration_mismatch": "Synchronize server/client HTML: ensure conditional renders match initial state, or use useEffect / suppressHydrationWarning.",
            "async_race_condition": "Use an abort controller (AbortController), atomic transactions, or debounce/lock to guard shared mutable state.",
            "null_undefined_access": "Add optional chaining (?.) and nullish coalescing (??) or explicit input validation before accessing nested attributes.",
            "cuda_oom": "Reduce batch size, enable torch.cuda.amp.autocast(dtype=torch.bfloat16), or free cache with torch.cuda.empty_cache().",
            "tensor_shape_mismatch": "Verify dimension compatibility across linear/attention projections and adjust with .view() or .reshape().",
            "gradient_nan_inf": "Implement torch.nn.utils.clip_grad_norm_, verify loss divisor is non-zero, or switch from fp16 to bfloat16."
        }
        recommended_fix = fix_actions.get(bug, "Inspect stack trace line numbers and check boundary conditions.")
        return {
            "bug_category": bug,
            "recommended_fix": recommended_fix,
            "real_inference_latency_ms": round(latency_ms, 2)
        }

    def triage_aiml_error(self, error_trace: str) -> Dict[str, Any]:
        ml_classes = {
            "cuda_oom": "GPU out of memory due to batch size or activation caching",
            "tensor_shape_mismatch": "Incompatible tensor dimensions during matmul or concat",
            "gradient_nan_inf": "Exploding gradients, NaN in loss, or numerical instability",
            "device_mismatch": "Tensors located on conflicting devices (CPU vs CUDA)",
            "dataloader_bottleneck": "Deadlock or slow multiprocessing in PyTorch DataLoader",
            "precision_bf16_error": "Unsupported dtype or lack of autocast context"
        }
        q_cause = {
            "t": "choice",
            "ins": "What is the primary root cause of this AI/ML training or inference failure?",
            "crit": ml_classes
        }
        probs_cause, latency_ms = self._infer_decision(error_trace, q_cause)
        pred_cause = list(ml_classes.keys())[torch.argmax(probs_cause).item()]

        q_vram = {
            "t": "score",
            "ins": "Rate how aggressive VRAM optimization must be to resolve this issue (0 = none needed, 4 = extreme).",
            "crit": {str(i): f"Level {i}" for i in range(5)}
        }
        probs_vram, _ = self._infer_decision(error_trace, q_vram)
        vram_score = int(torch.argmax(probs_vram).item())

        q_refactor = {
            "t": "noul",
            "ins": "Does resolving this ML issue require code architecture refactoring rather than just config/hyperparameter tuning?",
            "crit": None
        }
        probs_refactor, _ = self._infer_decision(error_trace, q_refactor)
        requires_refactor = bool(probs_refactor[1].item() >= 0.5)

        remediations = {
            "cuda_oom": "Enable gradient checkpointing, reduce batch size, or enable bfloat16 autocast.",
            "tensor_shape_mismatch": "Verify dimension shapes across layer forward passes; adjust .view() / .reshape().",
            "gradient_nan_inf": "Add torch.nn.utils.clip_grad_norm_, decrease learning rate, or verify loss epsilon.",
            "device_mismatch": "Call .to(device) on all incoming input batches and newly created sub-tensors.",
            "dataloader_bottleneck": "Tune num_workers (e.g. 2-4), set pin_memory=True, or use persistent_workers=True.",
            "precision_bf16_error": "Wrap forward pass with torch.autocast('cuda', dtype=torch.bfloat16)."
        }

        return {
            "ml_root_cause": pred_cause,
            "vram_mitigation_score": vram_score,
            "requires_code_refactor": requires_refactor,
            "recommended_fix": remediations.get(pred_cause, "Inspect traceback and verify model execution state."),
            "real_inference_latency_ms": round(latency_ms, 2)
        }

    def pick_best_option(
        self,
        context_or_problem: str,
        options: Dict[str, str],
        instruction: str = "Which of these proposed options is the best technical approach or fix?"
    ) -> Dict[str, Any]:
        """
        Dynamically evaluates an arbitrary set of candidate options using marker-token pooling.
        Returns ranked choices with calibrated confidence probabilities and safety gating.
        """
        if not options:
            return {"error": "Options dictionary cannot be empty."}

        q_choice = {
            "t": "choice",
            "ins": instruction,
            "crit": options
        }
        probs, latency_ms = self._infer_decision(context_or_problem, q_choice)

        ranked = []
        keys = list(options.keys())
        for idx, key in enumerate(keys):
            prob = float(probs[idx].item()) if idx < len(probs) else 0.0
            ranked.append({
                "option": key,
                "description": options[key],
                "confidence": round(prob, 4),
                "percentage": f"{prob:.1%}"
            })
        ranked.sort(key=lambda x: x["confidence"], reverse=True)
        best = ranked[0]

        eval_plan = f"Task Context: {context_or_problem}\nChosen Action ({best['option']}): {best['description']}"
        gate = self.should_autopilot(eval_plan)

        return {
            "best_option": best["option"],
            "best_description": best["description"],
            "best_confidence": best["confidence"],
            "best_percentage": best["percentage"],
            "ranking": ranked,
            "risk_score": gate.get("risk_score", 1),
            "allow_unsupervised": gate.get("allow_unsupervised", True),
            "route_tier": gate.get("route_tier", "fast_system1"),
            "real_inference_latency_ms": round(latency_ms, 2)
        }


# Backwards-compatible alias
RealLayaDecisionServer = CortexDecisionServer


def run_stdio_server():
    server = CortexDecisionServer()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            if method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {"name": "should_autopilot", "description": "Cortex-1: Real ModernBERT gating decision."},
                            {"name": "risk_score", "description": "Cortex-1: Real ModernBERT 0-4 risk scoring."},
                            {"name": "triage_security", "description": "Cortex-1: Real ModernBERT CWE vulnerability classifier."},
                            {"name": "diagnose_root_cause", "description": "Cortex-1: Real ModernBERT bug diagnosis."},
                            {"name": "triage_aiml_error", "description": "Cortex-1: Real ModernBERT AI/ML & PyTorch error triage."},
                            {"name": "pick_best_option", "description": "Cortex-1: Real ModernBERT dynamic option ranker & best choice picker."}
                        ]
                    }
                }
            elif method == "tools/call":
                tool = params.get("name")
                args = params.get("arguments", {})

                if tool == "should_autopilot":
                    res = server.should_autopilot(args.get("plan_text", ""))
                elif tool == "risk_score":
                    res = server.risk_score(args.get("code_diff_or_plan", ""))
                elif tool == "triage_security":
                    res = server.triage_security(args.get("code_snippet", ""), args.get("file_path", ""))
                elif tool == "diagnose_root_cause":
                    res = server.diagnose_root_cause(args.get("trace_or_snippet", ""))
                elif tool == "triage_aiml_error":
                    res = server.triage_aiml_error(args.get("error_trace", ""))
                elif tool == "pick_best_option":
                    res = server.pick_best_option(
                        args.get("context_or_problem", ""),
                        args.get("options", {}),
                        args.get("instruction", "Which of these proposed options is the best technical approach or fix?")
                    )
                else:
                    res = {"error": f"Unknown tool: {tool}"}

                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
                }
            else:
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"status": "ok"}}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
