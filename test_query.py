"""
Interactive Query Runner for Laya System 1 Decision Engine.
Allows testing ANY arbitrary user question, plan, diff, or error trace on RTX 5050.
"""

import sys
import json
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from cortex_mcp.server import CortexDecisionServer, RealLayaDecisionServer


def test_question(query: str, query_type: str = "auto"):
    """
    Test any query on Laya.
    query_type: 'gating', 'security', 'aiml', 'diagnostics', or 'all'
    """
    print(f"\n{'='*70}")
    print(f"INPUT QUERY:\n{query}")
    print(f"{'='*70}")

    engine = RealLayaDecisionServer()

    if query_type in ("gating", "all", "auto"):
        print("\n--- [1] AGENT AUTOPILOT GATING & RISK SCORING ---")
        t0 = time.perf_counter()
        gate_res = engine.should_autopilot(query)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  * Allow Unsupervised: {'PROCEED (True)' if gate_res['allow_unsupervised'] else 'STOP FOR USER REVIEW (False)'}")
        print(f"  * Confidence:         {gate_res['confidence']:.2%}")
        print(f"  * Risk Score:         {gate_res['risk_score']}/4")
        print(f"  * Route Tier:         {gate_res['route_tier']}")
        print(f"  * GPU Latency:        {gate_res['real_inference_latency_ms']:.2f} ms")

    if query_type in ("security", "all") or (query_type == "auto" and any(k in query.lower() for k in ["select", "query", "<script", "http", "token", "password", "exec", "os.system"])):
        print("\n--- [2] CYBERSECURITY & APPSEC TRIAGE ---")
        t0 = time.perf_counter()
        sec_res = engine.triage_security(query)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  * Vulnerability Class: {sec_res['vulnerability_class']}")
        print(f"  * Exploitability:      {sec_res['exploitability_score']}/4")
        print(f"  * Immediate Blocker:   {sec_res['is_immediate_blocker']}")
        print(f"  * Recommended Fix:     {sec_res['suggested_fix_strategy']}")
        print(f"  * GPU Latency:         {sec_res['real_inference_latency_ms']:.2f} ms")

    if query_type in ("aiml", "all") or (query_type == "auto" and any(k in query.lower() for k in ["cuda", "oom", "tensor", "shape", "nan", "torch", "gradient", "loss", "device"])):
        print("\n--- [3] AI / ML ENGINEERING TRIAGE ---")
        t0 = time.perf_counter()
        ml_res = engine.triage_aiml_error(query)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  * ML Root Cause:       {ml_res['ml_root_cause']}")
        print(f"  * VRAM Mitigation:     {ml_res['vram_mitigation_score']}/4")
        print(f"  * Requires Refactor:   {ml_res['requires_code_refactor']}")
        print(f"  * Recommended Fix:     {ml_res['recommended_fix']}")
        print(f"  * GPU Latency:         {ml_res['real_inference_latency_ms']:.2f} ms")

    if query_type in ("diagnostics", "all") or (query_type == "auto" and any(k in query.lower() for k in ["error", "exception", "failed", "traceback", "cannot read", "undefined"])):
        print("\n--- [4] FULL-STACK REPO BUG DIAGNOSTICS ---")
        t0 = time.perf_counter()
        bug_res = engine.diagnose_root_cause(query)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  * Bug Category:        {bug_res['bug_category']}")
        print(f"  * Recommended Fix:     {bug_res['recommended_fix']}")
        print(f"  * GPU Latency:         {bug_res['real_inference_latency_ms']:.2f} ms")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_arg = " ".join(sys.argv[1:])
        test_question(query_arg, query_type="all")
    else:
        # Default test
        test_question("Drop foreign key constraint on payments table and execute bulk migration.", query_type="all")
