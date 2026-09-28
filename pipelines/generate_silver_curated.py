"""
Curated Technical Decision Dataset Generator (Tier B - Silver Expansion).
Generates high-density, diverse, and hard/ambiguous technical decision samples
across Cybersecurity, Full-Stack Web, AI/ML Engineering, and Systems.
Follows strict schema validation to ensure 100% format consistency.
"""

from pathlib import Path
from typing import List
from schemas.primitives import DecisionSample, StructuredState, DomainType, DataTier
from schemas.questions_catalog import (
    CYBERSECURITY_QUESTIONS,
    FULLSTACK_QUESTIONS,
    AIML_QUESTIONS,
    AUTOPILOT_GATING_QUESTIONS
)


CURATED_EXPANSIONS = [
    # --- Cybersecurity: Modern Edge Cases & Ambiguities ---
    {
        "id": "sec_jwt_none_algorithm_bypass",
        "domain": DomainType.CYBERSECURITY,
        "questions": CYBERSECURITY_QUESTIONS,
        "title": "JWT Header alg: 'none' Signature Verification Bypass",
        "file": "middleware/auth.ts",
        "language": "typescript",
        "code": "const token = req.headers['authorization']?.split(' ')[1];\nconst decoded = jwt.decode(token); // Decoded without verifying algorithm\nreq.user = decoded;",
        "context": "Authentication middleware parses JWT payload using decode() rather than verify() with an explicit algorithm whitelist.",
        "answers": {
            "vulnerability_class": "crypto_secret_leak",
            "exploitability_score": 4,
            "is_immediate_blocker": True
        }
    },
    {
        "id": "sec_oauth_redirect_regex_bypass",
        "domain": DomainType.CYBERSECURITY,
        "questions": CYBERSECURITY_QUESTIONS,
        "title": "OAuth2 Redirect URI Loose Regex Validation",
        "file": "auth/oauth_handler.py",
        "language": "python",
        "code": "if re.match(r'https://example.com.*', redirect_uri):\n    return redirect(f\"{redirect_uri}?code={auth_code}\")",
        "context": "Regex lacks end-of-domain anchor, permitting redirects to 'https://example.com.attacker.com/steal'.",
        "answers": {
            "vulnerability_class": "authz_idor",
            "exploitability_score": 3,
            "is_immediate_blocker": True
        }
    },
    {
        "id": "sec_cors_wildcard_with_credentials",
        "domain": DomainType.CYBERSECURITY,
        "questions": CYBERSECURITY_QUESTIONS,
        "title": "Misconfigured CORS Header Reflected Origin",
        "file": "server.js",
        "language": "javascript",
        "code": "app.use((req, res, next) => {\n  res.header('Access-Control-Allow-Origin', req.headers.origin);\n  res.header('Access-Control-Allow-Credentials', 'true');\n  next();\n});",
        "context": "Reflecting arbitrary untrusted request origin combined with Allow-Credentials: true enables cross-site authenticated data theft.",
        "answers": {
            "vulnerability_class": "authz_idor",
            "exploitability_score": 3,
            "is_immediate_blocker": True
        }
    },
    {
        "id": "sec_hard_negative_prepared_statement",
        "domain": DomainType.CYBERSECURITY,
        "questions": CYBERSECURITY_QUESTIONS,
        "title": "Clean PostgreSQL Query via psycopg2 Parameters",
        "file": "db/queries.py",
        "language": "python",
        "code": "cursor.execute(\"SELECT id, balance FROM accounts WHERE user_id = %s;\", (user_id,))",
        "context": "Parameterized SQL execution passing parameters separately to the database engine.",
        "answers": {
            "vulnerability_class": "none_secure",
            "exploitability_score": 0,
            "is_immediate_blocker": False
        }
    },

    # --- AI / ML Engineering: Distributed & VRAM Edge Cases ---
    {
        "id": "ml_ddp_rank0_save_deadlock",
        "domain": DomainType.AI_ML_ENGINEERING,
        "questions": AIML_QUESTIONS,
        "title": "DistributedDataParallel Rank 0 Synchronization Deadlock",
        "file": "train_ddp.py",
        "language": "python",
        "code": "if rank == 0:\n    save_checkpoint(model, path)\n    torch.distributed.barrier() # Deadlock: other ranks never enter this branch",
        "context": "Only rank 0 calls barrier() while ranks 1-7 proceed to next epoch, halting training indefinitely.",
        "answers": {
            "ml_root_cause": "device_mismatch",
            "vram_mitigation_score": 0,
            "requires_code_refactor": True
        }
    },
    {
        "id": "ml_cuda_oom_activation_checkpointing_solution",
        "domain": DomainType.AI_ML_ENGINEERING,
        "questions": AIML_QUESTIONS,
        "title": "FlashAttention + Gradient Checkpointing on ModernBERT 421M",
        "file": "train_laya.py",
        "language": "python",
        "code": "model.gradient_checkpointing_enable()\nwith torch.cuda.amp.autocast(dtype=torch.bfloat16):\n    out = model(**inputs)",
        "context": "Peak VRAM reduced from 7.8GB to 4.2GB by recomputing activations during backward pass.",
        "answers": {
            "ml_root_cause": "cuda_oom",
            "vram_mitigation_score": 3,
            "requires_code_refactor": False
        }
    },

    # --- Full-Stack Web & Next.js App Router ---
    {
        "id": "web_server_action_client_closure",
        "domain": DomainType.FULLSTACK_WEB,
        "questions": FULLSTACK_QUESTIONS,
        "title": "Next.js Server Action Leaking Internal Context",
        "file": "actions/update_user.ts",
        "language": "typescript",
        "code": "'use server'\nexport async function updateUser(formData: FormData) {\n  const id = formData.get('id');\n  await db.user.update({ where: { id: String(id) }, data: { role: 'admin' } });\n}",
        "context": "Server action executes privileged database update without verifying session authentication or current user role.",
        "answers": {
            "bug_category": "network_cors_error",
            "breaking_change_risk": 4,
            "recommended_action": "revert_or_redesign"
        }
    },
    {
        "id": "web_state_direct_mutation_react",
        "domain": DomainType.FULLSTACK_WEB,
        "questions": FULLSTACK_QUESTIONS,
        "title": "React State Direct Array Mutation Without Re-render",
        "file": "components/TodoList.jsx",
        "language": "javascript",
        "code": "const [items, setItems] = useState(['Buy milk']);\nfunction addItem(item) {\n  items.push(item); // Direct array mutation\n  setItems(items);  // Same reference, React skips render\n}",
        "context": "Directly pushing into state array maintains the same object reference, preventing React reconciliation.",
        "answers": {
            "bug_category": "state_mutation",
            "breaking_change_risk": 1,
            "recommended_action": "revert_or_redesign"
        }
    },

    # --- Architecture & Autopilot Gating ---
    {
        "id": "adr_database_read_replica_routing",
        "domain": DomainType.AGENT_AUTOPILOT,
        "questions": AUTOPILOT_GATING_QUESTIONS,
        "title": "ADR-011: Implement Read/Write Database Connection Splitting",
        "file": "infra/database.py",
        "language": "python",
        "code": "def get_db_session(read_only: bool = False):\n    return replica_session() if read_only else primary_session()",
        "context": "Routes read-heavy analytics queries to AWS Aurora read replicas to reduce load on the primary writer.",
        "answers": {
            "should_autopilot": False,
            "risk_score": 2,
            "route_task": "balanced_agent"
        }
    },
    {
        "id": "adr_simple_docstring_and_readme_tweak",
        "domain": DomainType.AGENT_AUTOPILOT,
        "questions": AUTOPILOT_GATING_QUESTIONS,
        "title": "ADR-012: Update README API Documentation URLs",
        "file": "README.md",
        "language": "markdown",
        "code": "- Documentation: https://docs.myproject.io/v2",
        "context": "Correcting typo in documentation URL inside README.md file.",
        "answers": {
            "should_autopilot": True,
            "risk_score": 0,
            "route_task": "fast_system1"
        }
    }
]


def append_curated_expansions(stage1_path: Path):
    samples = []
    for item in CURATED_EXPANSIONS:
        state = StructuredState(
            title=item["title"],
            file_path=item["file"],
            language=item["language"],
            code_snippet=item["code"],
            context=item["context"],
            metadata={"source": "curated_technical_expansion"}
        )
        sample = DecisionSample(
            id=item["id"],
            domain=item["domain"],
            tier=DataTier.TIER_B_SILVER,
            state=state,
            questions=item["questions"],
            answers=item["answers"]
        )
        samples.append(sample)

    with open(stage1_path, "a", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")

    print(f"[INFO] Appended {len(samples)} curated technical samples to {stage1_path}")


if __name__ == "__main__":
    p = Path("data/stage1_train_silver.jsonl")
    append_curated_expansions(p)
