"""
Catalog of standardized typed question templates for Laya Decision Engine.
Ensures uniform label distributions, clear instructions, and standardized criteria.
"""

from schemas.primitives import TypedQuestion, QuestionType


# --- 1. Agent Autopilot Gating ---
AUTOPILOT_GATING_QUESTIONS = {
    "should_autopilot": TypedQuestion(
        type=QuestionType.NOUL,
        instructions="Should this task or plan run unsupervised in autopilot mode without stopping for user approval?"
    ),
    "risk_score": TypedQuestion(
        type=QuestionType.SCORE,
        instructions="Rate the architectural, data loss, or system regression risk of this change on a 0-4 scale.",
        criteria={
            "0": "Cosmetic, documentation, or trivial local variable change with zero side effects.",
            "1": "Low risk: localized logic change covered by tests or minor UI tweak.",
            "2": "Medium risk: modifies shared utility, database query, or non-critical API endpoint.",
            "3": "High risk: alters core data model, auth logic, external API contract, or dependency tree.",
            "4": "Critical risk: destructive migration, irreversible data deletion, or security-sensitive infrastructure."
        }
    ),
    "route_task": TypedQuestion(
        type=QuestionType.CHOICE,
        instructions="Which model or reasoning tier should handle this task?",
        criteria={
            "fast_system1": "Single-file edits, standard boilerplate, simple syntax fixes, or lint resolution.",
            "balanced_agent": "Standard feature implementation, multi-step debugging, or unit test creation.",
            "deep_reasoning": "Complex architectural design, subtle race conditions, distributed bugs, or security audits."
        }
    )
}


# --- 2. Cybersecurity & AppSec ---
CYBERSECURITY_QUESTIONS = {
    "vulnerability_class": TypedQuestion(
        type=QuestionType.CHOICE,
        instructions="Identify the primary CWE vulnerability class present in this code snippet or diff.",
        criteria={
            "sql_injection": "Unsanitized input concatenated into database query (CWE-89).",
            "xss": "Unescaped user input rendered in web browser / HTML context (CWE-79).",
            "ssrf": "Server making outbound network request based on untrusted user URL (CWE-918).",
            "authz_idor": "Missing access control, broken object level authorization, or privilege escalation (CWE-285/639).",
            "command_injection": "Untrusted input passed to OS shell or process execution (CWE-78).",
            "crypto_secret_leak": "Hardcoded API keys, passwords, or insecure cryptographic algorithms (CWE-798/327).",
            "none_secure": "Code follows defensive security best practices; no exploitable vulnerability found."
        }
    ),
    "exploitability_score": TypedQuestion(
        type=QuestionType.SCORE,
        instructions="Rate the exploitability and impact of the vulnerability from 0 to 4.",
        criteria={
            "0": "No security impact / secure implementation.",
            "1": "Low: requires authenticated local access or unusual preconditions.",
            "2": "Medium: requires specific user interaction (e.g. CSRF/reflected XSS).",
            "3": "High: unauthenticated remote exploitation with data leakage.",
            "4": "Critical: unauthenticated remote code execution (RCE) or complete database compromise."
        }
    ),
    "is_immediate_blocker": TypedQuestion(
        type=QuestionType.NOUL,
        instructions="Must this security flaw block deployment or code commit immediately?"
    )
}


# --- 3. Full-Stack Web & Bugs ---
FULLSTACK_QUESTIONS = {
    "bug_category": TypedQuestion(
        type=QuestionType.CHOICE,
        instructions="What is the root cause classification of this web or application error?",
        criteria={
            "hydration_mismatch": "SSR markup does not match client DOM render tree (React/Next.js/Vue).",
            "async_race_condition": "Out-of-order promise resolution, stale closures, or unhandled useEffect cleanup.",
            "null_undefined_access": "Attempting to access properties of null/undefined or missing optional chaining.",
            "network_cors_error": "CORS preflight failure, missing headers, or 4xx/5xx API contract breakdown.",
            "state_mutation": "Direct mutation of immutable state causing missing re-renders.",
            "syntax_type_error": "TypeScript compilation error or invalid JavaScript syntax."
        }
    ),
    "breaking_change_risk": TypedQuestion(
        type=QuestionType.SCORE,
        instructions="Rate the likelihood and impact of breaking existing frontend or API consumers.",
        criteria={
            "0": "Zero risk: internal implementation detail with no API signature change.",
            "1": "Low risk: backward-compatible optional prop or parameter added.",
            "2": "Medium risk: deprecation of prop or slight change in component markup structure.",
            "3": "High risk: changed required API field or removed public hook/component.",
            "4": "Critical risk: widespread breaking change across core routing or authentication."
        }
    ),
    "recommended_action": TypedQuestion(
        type=QuestionType.CHOICE,
        instructions="What is the most effective immediate fix strategy?",
        criteria={
            "add_optional_chaining": "Use '?.' or nullish coalescing '??' to safely guard property access.",
            "fix_async_handling": "Wrap with async/await, abort controller, or proper promise sequencing.",
            "adjust_ssr_boundary": "Use 'use client', dynamic import with ssr: false, or useEffect guard.",
            "update_type_definition": "Refine TypeScript interface, types, or Zod schema.",
            "revert_or_redesign": "Revert change or redesign API interface."
        }
    )
}


# --- 4. AI / ML Engineering & Runtime ---
AIML_QUESTIONS = {
    "ml_root_cause": TypedQuestion(
        type=QuestionType.CHOICE,
        instructions="What is the primary root cause of this AI/ML training or inference failure?",
        criteria={
            "cuda_oom": "GPU out of memory due to batch size, unreleased cache, or activation memory.",
            "tensor_shape_mismatch": "Incompatible tensor dimensions during matrix multiplication, concat, or loss calc.",
            "gradient_nan_inf": "Exploding gradients, division by zero, or numerical instability in fp16.",
            "device_mismatch": "Attempting operations on tensors on different devices (CPU vs CUDA:0).",
            "dataloader_bottleneck": "Slow CPU multiprocessing, deadlocks in num_workers, or I/O bottleneck.",
            "precision_bf16_error": "Unsupported dtype on hardware or lack of autocast context."
        }
    ),
    "vram_mitigation_score": TypedQuestion(
        type=QuestionType.SCORE,
        instructions="Rate how aggressive VRAM optimization must be to resolve this issue (0 = none needed, 4 = extreme).",
        criteria={
            "0": "No VRAM issue present.",
            "1": "Minor: clear empty cache with torch.cuda.empty_cache().",
            "2": "Moderate: enable bf16 mixed precision or reduce batch size by 50%.",
            "3": "Significant: enable gradient checkpointing and gradient accumulation.",
            "4": "Extreme: apply CPU offloading (DeepSpeed ZeRO-3 / FSDP) or LoRA/QLoRA 4-bit."
        }
    ),
    "requires_code_refactor": TypedQuestion(
        type=QuestionType.NOUL,
        instructions="Does resolving this ML issue require code architecture refactoring rather than just config/hyperparameter tuning?"
    )
}
