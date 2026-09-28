"""
Laya Local Terminal Decision & Option-Ranking Engine.
Default Mode: Automatically ranks candidate options and recommends the optimal technical choice.
Runs 100% on-device on NVIDIA RTX 5050 Laptop GPU with sub-40ms neural forward pass.
"""

import sys
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from laya_mcp.server import RealLayaDecisionServer


# Standard candidate option pools for zero-option inputs
DOMAIN_DEFAULT_CANDIDATES = {
    "aiml": {
        "Option 1": "Enable gradient checkpointing and bfloat16 mixed precision autocast.",
        "Option 2": "Reduce per-device micro-batch size and increase gradient accumulation steps.",
        "Option 3": "Tune DataLoader num_workers (2-4), set pin_memory=True, and persistent_workers=True.",
        "Option 4": "Offload optimizer states or model parameters to CPU/NVMe via DeepSpeed ZeRO-3 / FSDP.",
        "Option 5": "Call torch.cuda.empty_cache() and detach intermediate loss calculation tensors."
    },
    "security": {
        "Option 1": "Use parameterized query bindings or ORM prepared statements; avoid string interpolation.",
        "Option 2": "Enforce server-side tenancy authorization checks (verify current_user owns target resource ID).",
        "Option 3": "Pass execution arguments as an array to subprocess without shell=True; avoid os.system.",
        "Option 4": "Strict domain and IP allowlisting; validate URLs and block RFC1918 internal metadata addresses.",
        "Option 5": "Rotate exposed credentials immediately and load secrets strictly from environment variables."
    },
    "diagnostics": {
        "Option 1": "Add optional chaining (?.) and nullish coalescing (??) before property access.",
        "Option 2": "Wrap with AbortController, atomic database transactions, or async promise sequencing.",
        "Option 3": "Adjust SSR boundaries using 'use client', dynamic import with ssr:false, or useEffect.",
        "Option 4": "Refactor component state updates to use immutable patterns without direct mutation.",
        "Option 5": "Refine TypeScript interface types and add runtime schema validation with Zod."
    },
    "gating": {
        "Option 1": "Execute unblocked in fast_system1 mode (isolated low-risk change).",
        "Option 2": "Split migration into backward-compatible phased PRs with automated rollback testing.",
        "Option 3": "Halt autonomous execution and require senior engineer manual peer review."
    }
}


def extract_options_from_text(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Extracts candidate options from user input.
    Supports:
    - Multi-line lists: Option A: ..., A) ..., 1. ..., - ...
    - Single-line comparisons: 'X vs Y vs Z' or 'X or Y'
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    context_lines = []
    options: Dict[str, str] = {}

    opt_pattern = re.compile(r"^(?:option\s+([a-zA-Z0-9]+)|([a-zA-Z0-9]+)[\.\)\:]|\-\s+option\s+([a-zA-Z0-9]+))\s*[:\-]?\s*(.*)$", re.IGNORECASE)

    for line in lines:
        m = opt_pattern.match(line)
        if m:
            key_name = m.group(1) or m.group(2) or m.group(3)
            opt_key = f"Option {key_name.upper()}"
            opt_desc = m.group(4).strip() if m.group(4) else line
            options[opt_key] = opt_desc
        else:
            if not options:
                context_lines.append(line)

    # Check for single-line ' vs ' pattern
    if not options and len(lines) == 1 and " vs " in lines[0].lower():
        parts = re.split(r"\s+vs\.?\s+", lines[0], flags=re.IGNORECASE)
        if len(parts) >= 2:
            context = "Compare and rank candidate approaches."
            for idx, part in enumerate(parts):
                letter = chr(65 + idx)
                options[f"Option {letter}"] = part.strip()
            return context, options

    context = "\n".join(context_lines) if context_lines else text
    return context, options


def detect_domain_intent(text: str) -> str:
    """Heuristic domain classifier to select primary triage head."""
    t_lower = text.lower()

    aiml_keywords = [
        "cuda", "oom", "out of memory", "tensor", "shape", "nan", "inf",
        "torch", "gradient", "loss", "device", "bf16", "fp16", "backprop",
        "backward()", "vram", "dataloader", "ddp", "fsdp", "autocast"
    ]
    if any(k in t_lower for k in aiml_keywords):
        return "aiml"

    sec_keywords = [
        "select", "union", "<script", "exec(", "os.system", "subprocess",
        "eval(", "passwd", "token", "secret", "private_key", "bearer",
        "idor", "ssrf", "injection", "vulnerability", "cwe", "jwt",
        "169.254.169.254", "authorization", "sanitize", "deserialize"
    ]
    if any(k in t_lower for k in sec_keywords):
        return "security"

    diag_keywords = [
        "traceback", "exception", "hydration", "race condition", "nullpointer",
        "undefined is not", "keyerror", "500 internal", "syntaxerror",
        "typeerror", "cannot read property", "unhandled rejection"
    ]
    if any(k in t_lower for k in diag_keywords):
        return "diagnostics"

    return "gating"


def format_ranking_response(
    domain_label: str,
    context: str,
    ranked_result: Dict[str, Any],
    latency_ms: float
) -> str:
    """Formats decision ranking and options into clean enterprise dialogue."""
    lines = [f"\n>>> Cortex-1 ({latency_ms:.1f}ms):"]
    lines.append(f"    [Domain]            : {domain_label}")
    lines.append(f"    [Best Option]       : {ranked_result.get('best_option')} - {ranked_result.get('best_description')} [RECOMMENDED]")
    lines.append(f"    [Confidence]        : {ranked_result.get('best_percentage', '0.0%')}")
    lines.append("    [Ranked Candidates] :")

    for r in ranked_result.get("ranking", []):
        is_winner = r["option"] == ranked_result.get("best_option")
        badge = " [RECOMMENDED]" if is_winner else ""
        lines.append(f"      - {r['option']}: {r['description']} [{r['percentage']} confidence]{badge}")

    status = "PROCEED WITH BEST OPTION" if ranked_result.get("allow_unsupervised") else "HALT FOR REVIEW BEFORE EXECUTION"
    lines.append(f"    [Autopilot Gating]  : {status} (Risk {ranked_result.get('risk_score', 1)}/4)")
    lines.append(f"    [Recommended Route] : {ranked_result.get('route_tier', 'fast_system1')}")
    lines.append("")
    return "\n".join(lines)


def interactive_wizard(server: RealLayaDecisionServer):
    """Explicit step-by-step wizard if user requests /wizard."""
    print("\n--- [DECISION & OPTION RANKING WIZARD] ---")
    print("Enter the dilemma context, then enter 2 to 5 candidate options.")
    context = input("Enter Context / Problem: ").strip()
    if not context:
        context = "Evaluate the optimal engineering approach."

    options: Dict[str, str] = {}
    opt_idx = 1
    print("\nEnter options (leave empty and press Enter when done):")
    while opt_idx <= 5:
        letter = chr(64 + opt_idx)
        opt_text = input(f"Option {letter}: ").strip()
        if not opt_text:
            if opt_idx <= 2:
                print("Please enter at least 2 options to compare.")
                continue
            break
        options[f"Option {letter}"] = opt_text
        opt_idx += 1

    t_start = time.perf_counter()
    res = server.pick_best_option(context, options)
    latency_ms = (time.perf_counter() - t_start) * 1000

    print(format_ranking_response("Candidate Architecture & Decision Ranking", context, res, latency_ms))


def run_chat_session():
    print("\n" + "=" * 76)
    print("        CORTEX-1 LARGE: DEFAULT OPTION-RANKING DECISION COPILOT")
    print("        Model: Cortex-1 Large (421M ModernBERT Backbone) | RTX 5050 (sm_120)")
    print("        Mode: Option Ranking is DEFAULT. Enter any dilemma or choices.")
    print("=" * 76)
    print("Usage:")
    print("  * Type a problem with candidate options (e.g. 'Option A: ... Option B: ...')")
    print("  * Compare choices inline using 'vs' (e.g. 'Redis cache vs PostgreSQL query')")
    print("  * Paste any bug, traceback, or code diff to get the ranked fix actions.")
    print("  * Commands: '/wizard' for step-by-step | '/paste' for multi-line | '/exit' to quit.")
    print("-" * 76)

    server = RealLayaDecisionServer()

    print("\n[READY] Cortex-1 Large resident in GPU VRAM. Enter problem and options below:\n")

    while True:
        try:
            user_input = input(">>> You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Cortex-1 Chat. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("Session terminated.")
            break

        if cmd in ("/wizard", "/pick", "/choose"):
            interactive_wizard(server)
            continue

        if cmd == "/help":
            print("\n[EXAMPLE QUERIES]:")
            print("  1. Custom Options:")
            print("     Django session permissions.")
            print("     Option A: Cache in client JWT cookie")
            print("     Option B: Cache in Redis with 15min TTL and DB fallback")
            print("  2. Inline Comparison:")
            print("     Redis cluster vs Kafka consumer queue vs PostgreSQL table")
            print("  3. Direct Error / Triage:")
            print("     'PyTorch training loop runs out of memory on step 40 when sequence length is 4096.'\n")
            continue

        if cmd in ("/paste", "/multi"):
            print("[PASTE MODE] Paste code / proposal lines. Type 'END' on a new line when done:")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            user_input = "\n".join(lines).strip()
            if not user_input:
                continue

        context, options = extract_options_from_text(user_input)
        domain = detect_domain_intent(user_input)
        t_start = time.perf_counter()

        # If user did not provide explicit options, supply the domain's candidate solution pool
        if len(options) < 2:
            options = DOMAIN_DEFAULT_CANDIDATES.get(domain, DOMAIN_DEFAULT_CANDIDATES["gating"])

        domain_titles = {
            "aiml": "AI / ML Runtime Triage & Fix Ranking",
            "security": "Cybersecurity & Remediation Ranking",
            "diagnostics": "Full-Stack Bug Diagnosis & Fix Ranking",
            "gating": "Architecture & Pull Request Decision Ranking"
        }
        domain_label = domain_titles.get(domain, "Architecture Decision Ranking")

        res_pick = server.pick_best_option(context, options)
        total_latency = (time.perf_counter() - t_start) * 1000

        print(format_ranking_response(domain_label, context, res_pick, total_latency))


if __name__ == "__main__":
    run_chat_session()
