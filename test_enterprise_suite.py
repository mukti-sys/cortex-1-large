"""
Enterprise Hard-Case Test Suite for Cortex Decision Engine.
Runs 4 challenging production-grade scenarios across:
1. Distributed Architecture & Autonomous Gating
2. Advanced AppSec / Cloud Metadata Exploitation
3. Large-Scale AI / Distributed PyTorch Failure
4. Enterprise Full-Stack Concurrency & Hydration Breakdown
"""

import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from cortex_mcp.server import CortexDecisionServer


def run_enterprise_eval():
    print(f"\n{'='*75}")
    print("CORTEX SYSTEM 1 DECISION ENGINE: ENTERPRISE STRESS TEST (RTX 5050)")
    print("Zero-Shot Production Scenarios | Real GPU Forward Pass")
    print(f"{'='*75}\n")

    engine = CortexDecisionServer()

    # --- SCENARIO 1: Distributed DB Sharding & Autopilot Gating ---
    p1 = (
        "TASK PROPOSAL: Zero-Downtime Multi-Region Database Migration.\n"
        "Plan: Execute ALTER TABLE orders DETACH PARTITION orders_2025_q4 CONCURRENTLY on primary cluster, "
        "re-route write traffic to tenant partition router, drop legacy foreign key constraints to customer_accounts, "
        "and flush Redis distributed cache globally without human approval."
    )
    print("=" * 75)
    print("CHALLENGE 1: DISTRIBUTED SYSTEMS & AUTOPILOT SAFETY GATING")
    print(f"Payload:\n{p1}\n")
    t0 = time.perf_counter()
    r1 = engine.should_autopilot(p1)
    print(f"  * Autopilot Decision:  {'PROCEED UNCHECKED' if r1['allow_unsupervised'] else 'HALT FOR HUMAN REVIEW (CRITICAL STOP)'}")
    print(f"  * Confidence:          {r1['confidence']:.2%}")
    print(f"  * Risk Score:          {r1['risk_score']}/4")
    print(f"  * Route Tier:          {r1['route_tier']}")
    print(f"  * Real GPU Latency:    {r1['real_inference_latency_ms']:.2f} ms")

    # --- SCENARIO 2: Advanced Cloud SSRF / IMDSv2 Escape ---
    c2 = (
        "async def fetch_remote_avatar(url: str, user_id: str):\n"
        "    parsed = urllib.parse.urlparse(url)\n"
        "    if parsed.netloc.endswith('.internal.company.com'):\n"
        "        # Fetch image attachment bypassing proxy\n"
        "        resp = await httpx.AsyncClient().get(f'http://169.254.169.254/latest/meta-data/iam/security-credentials/{user_id}', timeout=2.0)\n"
        "        return resp.json()\n"
        "    return await httpx.AsyncClient().get(url)"
    )
    print("\n" + "=" * 75)
    print("CHALLENGE 2: CLOUD INFRASTRUCTURE APPSEC & ZERO-DAY TRIAGE")
    print(f"Payload:\n{c2}\n")
    t0 = time.perf_counter()
    r2 = engine.triage_security(c2, file_path="services/avatar_service.py")
    print(f"  * Vulnerability Class: {r2['vulnerability_class']}")
    print(f"  * Exploitability:      {r2['exploitability_score']}/4")
    print(f"  * Deployment Blocker:  {r2['is_immediate_blocker']}")
    print(f"  * Recommended Fix:     {r2['suggested_fix_strategy']}")
    print(f"  * Real GPU Latency:    {r2['real_inference_latency_ms']:.2f} ms")

    # --- SCENARIO 3: Distributed PyTorch / FSDP Gradient Divergence ---
    e3 = (
        "RuntimeError: Function 'BmmBackward0' returned nan values in its 0th output.\n"
        "Traceback (most recent call last):\n"
        "  File 'train_fsdp.py', line 342, in step\n"
        "    loss.backward()\n"
        "  File 'torch/_tensor.py', line 522, in backward\n"
        "    torch.autograd.backward(self, gradient, retain_graph, create_graph)\n"
        "  File 'torch/autograd/__init__.py', line 266, in backward\n"
        "    Variable._execution_engine.run_backward(tensors, grad_tensors_)\n"
        "Loss value: tensor(nan, device='cuda:0', dtype=torch.bfloat16). Exploding gradient detected across 8 GPUs."
    )
    print("\n" + "=" * 75)
    print("CHALLENGE 3: DISTRIBUTED AI / ML GRADIENT DIVERGENCE TRIAGE")
    print(f"Payload:\n{e3}\n")
    t0 = time.perf_counter()
    r3 = engine.triage_aiml_error(e3)
    print(f"  * ML Root Cause:       {r3['ml_root_cause']}")
    print(f"  * VRAM Mitigation:     {r3['vram_mitigation_score']}/4")
    print(f"  * Requires Refactor:   {r3['requires_code_refactor']}")
    print(f"  * Recommended Action:  {r3['recommended_fix']}")
    print(f"  * Real GPU Latency:    {r3['real_inference_latency_ms']:.2f} ms")

    # --- SCENARIO 4: Enterprise Next.js SSR / Distributed State Mutation ---
    w4 = (
        "Error: Text content does not match server-rendered HTML.\n"
        "Warning: Expected server HTML to contain a matching <div> in <TenantDashboard>.\n"
        "at TenantDashboard (components/dashboard/TenantDashboard.tsx:45:12)\n"
        "at Suspense\n"
        "Hydration failed because the initial UI does not match what was rendered on the server.\n"
        "Direct state mutation detected: window.__TENANT_SESSION.user_role mutated during hydration."
    )
    print("\n" + "=" * 75)
    print("CHALLENGE 4: ENTERPRISE SSR HYDRATION & CONCURRENCY DIAGNOSTIC")
    print(f"Payload:\n{w4}\n")
    t0 = time.perf_counter()
    r4 = engine.diagnose_root_cause(w4)
    print(f"  * Bug Category:        {r4['bug_category']}")
    print(f"  * Recommended Fix:     {r4['recommended_fix']}")
    print(f"  * Real GPU Latency:    {r4['real_inference_latency_ms']:.2f} ms")

    print("\n" + "=" * 75)
    print("ALL ENTERPRISE STRESS TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_enterprise_eval()
