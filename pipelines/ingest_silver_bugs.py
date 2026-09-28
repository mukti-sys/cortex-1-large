"""
Full-Stack Web & AI/ML Bug Diagnostics Dataset Generator (Tier B - Silver).
Generates 100+ diagnostic cases covering React hydration, async race conditions,
PyTorch CUDA OOM, and tensor dimension mismatches.
"""

from pathlib import Path
from typing import List
from schemas.primitives import DecisionSample, StructuredState, DomainType, DataTier
from schemas.questions_catalog import FULLSTACK_QUESTIONS, AIML_QUESTIONS


BUG_PATTERNS = [
    # Full-Stack Web
    ("hydration_date", DomainType.FULLSTACK_WEB, FULLSTACK_QUESTIONS,
     "components/LiveTime.tsx", "typescript",
     "export default function LiveTime() { return <div>{new Date().toLocaleTimeString()}</div>; }",
     "Error: Hydration failed because server HTML did not match client render.",
     "Client date evaluates differently on server vs browser.",
     {"bug_category": "hydration_mismatch", "breaking_change_risk": 1, "recommended_action": "adjust_ssr_boundary"}),

    ("hydration_localstorage", DomainType.FULLSTACK_WEB, FULLSTACK_QUESTIONS,
     "components/ThemeSwitch.tsx", "typescript",
     "const theme = window.localStorage.getItem('theme') || 'dark';\nreturn <body className={theme}>{children}</body>;",
     "ReferenceError: window is not defined during SSR / hydration mismatch.",
     "Directly reading window object during server component rendering.",
     {"bug_category": "hydration_mismatch", "breaking_change_risk": 2, "recommended_action": "adjust_ssr_boundary"}),

    ("async_race_fetch", DomainType.FULLSTACK_WEB, FULLSTACK_QUESTIONS,
     "hooks/useTypeahead.ts", "typescript",
     "useEffect(() => { fetch(`/search?q=${q}`).then(r => r.json()).then(setList); }, [q]);",
     "Stale results display when typing fast.",
     "Previous slow query resolves after newer fast query.",
     {"bug_category": "async_race_condition", "breaking_change_risk": 2, "recommended_action": "fix_async_handling"}),

    ("null_prop_access", DomainType.FULLSTACK_WEB, FULLSTACK_QUESTIONS,
     "views/AccountView.jsx", "javascript",
     "return <div>Billing email: {account.billingInfo.contact.email}</div>;",
     "TypeError: Cannot read properties of undefined (reading 'contact')",
     "Accessing nested property without nullish guard or optional chaining.",
     {"bug_category": "null_undefined_access", "breaking_change_risk": 1, "recommended_action": "add_optional_chaining"}),

    ("state_mutation_sort", DomainType.FULLSTACK_WEB, FULLSTACK_QUESTIONS,
     "components/Leaderboard.tsx", "typescript",
     "const sorted = users.sort((a, b) => b.score - a.score);\nsetUsers(sorted);",
     "List does not re-render upon sorting.",
     "Array.prototype.sort mutates array in place; reference is unchanged.",
     {"bug_category": "state_mutation", "breaking_change_risk": 1, "recommended_action": "revert_or_redesign"}),

    # AI / ML Engineering
    ("cuda_oom_backward", DomainType.AI_ML_ENGINEERING, AIML_QUESTIONS,
     "train.py", "python",
     "outputs = model(**inputs)\nloss = outputs.loss\nloss.backward()",
     "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.80 GiB on GPU 0",
     "Activation memory exceeds 8GB VRAM with batch size 16 and seq_len 2048.",
     {"ml_root_cause": "cuda_oom", "vram_mitigation_score": 3, "requires_code_refactor": False}),

    ("tensor_shape_matmul", DomainType.AI_ML_ENGINEERING, AIML_QUESTIONS,
     "layers/attention.py", "python",
     "scores = torch.matmul(query, key) # query: (B, H, S, D), key: (B, H, S, D)",
     "RuntimeError: The size of tensor a (128) must match the size of tensor b (64) at non-singleton dimension 3",
     "Missing transpose on key tensor (key.transpose(-1, -2)).",
     {"ml_root_cause": "tensor_shape_mismatch", "vram_mitigation_score": 0, "requires_code_refactor": True}),

    ("fp16_nan_underflow", DomainType.AI_ML_ENGINEERING, AIML_QUESTIONS,
     "train_amp.py", "python",
     "with torch.cuda.amp.autocast(dtype=torch.float16):\n    loss = model(x).loss\nloss.backward()",
     "Warning: Loss is NaN, skipping optimizer step.",
     "FP16 exponent range underflows during backprop without GradScaler.",
     {"ml_root_cause": "gradient_nan_inf", "vram_mitigation_score": 2, "requires_code_refactor": False})
]


def generate_bugs_dataset(output_path: Path, target_count: int = 100) -> List[DecisionSample]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples: List[DecisionSample] = []
    idx = 0

    while len(samples) < target_count:
        for prefix, domain, questions, fpath, lang, code, trace, ctx, answers in BUG_PATTERNS:
            idx += 1
            iteration = (idx // len(BUG_PATTERNS)) + 1
            sample_id = f"tier_b_bug_{prefix}_{idx:04d}"

            state = StructuredState(
                title=f"Bug Diagnostic #{idx}: {prefix.replace('_', ' ').title()}",
                file_path=fpath,
                language=lang,
                code_snippet=code,
                stack_trace=trace,
                context=f"{ctx} (Trace diagnostic #{idx})",
                metadata={"domain": domain.value, "iteration": iteration}
            )

            sample = DecisionSample(
                id=sample_id,
                domain=domain,
                tier=DataTier.TIER_B_SILVER,
                state=state,
                questions=questions,
                answers=answers
            )
            samples.append(sample)
            if len(samples) >= target_count:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")

    print(f"[INFO] Generated {len(samples)} Tier B Bug samples to {output_path}")
    return samples


if __name__ == "__main__":
    generate_bugs_dataset(Path("data/tier_b_silver/bugs_diagnostics.jsonl"), target_count=100)
