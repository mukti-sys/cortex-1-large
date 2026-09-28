"""
Test Client for Laya Decision Engine MCP Server.
Sends mock queries for:
1. should_autopilot
2. risk_score
3. triage_security
4. diagnose_root_cause
And measures latency against the 35ms target.
"""

import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from laya_mcp.server import RealLayaDecisionServer


def run_tests():
    print(f"\n{'='*65}")
    print("LAYA SYSTEM 1 DECISION ENGINE: REAL TRANSFORMER BENCHMARK")
    print(f"{'='*65}")

    engine = RealLayaDecisionServer()

    # Benchmark 1: Pure Single-Decision Primitive (risk_score - SCORE)
    code_diff = "def update_user(id, data):\n-   db.execute('UPDATE users SET ...')\n+   db.table('users').where('id', id).update(data)"
    t0 = time.perf_counter()
    res1 = engine.risk_score(code_diff)
    dt1 = (time.perf_counter() - t0) * 1000

    print("[TEST 1: ATOMIC PRIMITIVE] risk_score (SCORE 0-4):")
    print(f"  * Code Diff:   '{code_diff.splitlines()[0]}...'")
    print(f"  * Score:       {res1['score']}/4 ({res1['rubric_level']})")
    print(f"  * Critical:    {res1['is_critical']} (Confidence: {res1['confidence']:.2%})")
    print(f"  * GPU Latency: {res1['real_inference_latency_ms']:.2f} ms")
    print("-" * 65)

    # Benchmark 2: Pure Single-Decision Primitive (diagnose_root_cause - CHOICE)
    stack_trace = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.40 GiB on GPU 0"
    t0 = time.perf_counter()
    res2 = engine.diagnose_root_cause(stack_trace)
    dt2 = (time.perf_counter() - t0) * 1000

    print("[TEST 2: ATOMIC PRIMITIVE] diagnose_root_cause (CHOICE):")
    print(f"  * Trace:       {stack_trace}")
    print(f"  * Bug Type:    {res2['bug_category']}")
    print(f"  * Fix Action:  {res2['recommended_fix']}")
    print(f"  * GPU Latency: {res2['real_inference_latency_ms']:.2f} ms")
    print("-" * 65)

    # Benchmark 3: Gating Compound Endpoint (should_autopilot - NOUL + SCORE)
    plan_text = "Refactor database migration to drop column 'legacy_token' and re-index customer table."
    t0 = time.perf_counter()
    res3 = engine.should_autopilot(plan_text)
    dt3 = (time.perf_counter() - t0) * 1000

    print("[TEST 3: COMPOUND ENDPOINT] should_autopilot (NOUL + SCORE):")
    print(f"  * Plan:        '{plan_text[:60]}...'")
    print(f"  * Autopilot:   {res3['allow_unsupervised']} (Risk Score: {res3['risk_score']}/4)")
    print(f"  * Route Tier:  {res3['route_tier']}")
    print(f"  * Single GPU:  {res3['real_inference_latency_ms']:.2f} ms per decision pass")
    print(f"  * Wall Clock:  {dt3:.2f} ms (dual decision)")
    print("-" * 65)

    # Benchmark 4: Security Compound Endpoint (triage_security - CWE + SCORE + NOUL)
    vulnerable_sql = "query = f\"SELECT * FROM users WHERE email = '{user_input}'\""
    t0 = time.perf_counter()
    res4 = engine.triage_security(vulnerable_sql)
    dt4 = (time.perf_counter() - t0) * 1000

    print("[TEST 4: COMPOUND ENDPOINT] triage_security (CWE + SCORE + NOUL):")
    print(f"  * Code:        {vulnerable_sql}")
    print(f"  * Vuln Class:  {res4['vulnerability_class']}")
    print(f"  * Severity:    {res4['exploitability_score']}/4 (Blocker: {res4['is_immediate_blocker']})")
    print(f"  * Single GPU:  {res4['real_inference_latency_ms']:.2f} ms per decision pass")
    print(f"  * Wall Clock:  {dt4:.2f} ms (triple decision)")
    # Benchmark 5: AI / ML Runtime Triage (triage_aiml_error - CHOICE + SCORE + NOUL)
    oom_trace = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB on GPU 0. Reserved memory: 7.50 GiB"
    t0 = time.perf_counter()
    res5 = engine.triage_aiml_error(oom_trace)
    dt5 = (time.perf_counter() - t0) * 1000

    print("[TEST 5: COMPOUND ENDPOINT] triage_aiml_error (CHOICE + SCORE + NOUL):")
    print(f"  * Trace:       {oom_trace}")
    print(f"  * Root Cause:  {res5['ml_root_cause']}")
    print(f"  * VRAM Score:  {res5['vram_mitigation_score']}/4 (Needs Refactor: {res5['requires_code_refactor']})")
    print(f"  * Suggested:   {res5['recommended_fix']}")
    print(f"  * Single GPU:  {res5['real_inference_latency_ms']:.2f} ms per decision pass")
    print(f"  * Wall Clock:  {dt5:.2f} ms (triple decision)")
    print("=" * 65)

    atomic_latencies = [res1['real_inference_latency_ms'], res2['real_inference_latency_ms']]
    avg_atomic_gpu = sum(atomic_latencies) / len(atomic_latencies)
    print(f"[SUMMARY] Mean Atomic Single-Decision GPU Latency: {avg_atomic_gpu:.2f} ms (Target: < 35.0 ms) -> [{'PASS' if avg_atomic_gpu < 35.0 else 'FAIL'}]")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_tests()
