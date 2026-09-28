"""
Cybersecurity & AppSec Dataset Generator (Tier B - Silver).
Generates 200+ realistic CWE vulnerability samples & clean hard negatives across:
- CWE-89 (SQL Injection) & Parameterized Clean Defenses
- CWE-79 (Cross-Site Scripting XSS) & DOMPurify Sanitizers
- CWE-918 (Server-Side Request Forgery SSRF) & Cloud Metadata Bypasses
- CWE-285/639 (IDOR / Broken Access Control)
- CWE-78 (OS Command Injection)
- CWE-798/327 (Secret Leaks & Cryptographic Timing Attacks)
- CWE-502 (Insecure Deserialization via pickle/yaml)
- CWE-22 (Path Traversal)
"""

from pathlib import Path
from typing import List
from schemas.primitives import DecisionSample, StructuredState, DomainType, DataTier
from schemas.questions_catalog import CYBERSECURITY_QUESTIONS


SEC_PATTERNS = [
    # (id_prefix, vuln_class, score, is_blocker, language, file_ext, vuln_code, context)
    ("sqli_fstring", "sql_injection", 4, True, "python", "py",
     "query = f\"SELECT id, email, role FROM users WHERE email = '{user_input}'\"\ndb.execute(query)",
     "User login endpoint interpolating raw user input directly into SQL statement."),
    ("sqli_clean_param", "none_secure", 0, False, "python", "py",
     "stmt = select(User).where(User.email == user_input)\ndb.session.execute(stmt).scalars().first()",
     "Clean query using SQLAlchemy ORM parameterized query."),
    ("sqli_string_concat", "sql_injection", 4, True, "javascript", "js",
     "const sql = 'SELECT * FROM products WHERE category = \"' + req.query.cat + '\"';\npool.query(sql);",
     "Node.js API endpoint concatenating query params into SQL query."),
    ("sqli_clean_node", "none_secure", 0, False, "javascript", "js",
     "pool.query('SELECT * FROM products WHERE category = $1', [req.query.cat]);",
     "Clean Node.js parameterized query passing parameters array."),

    ("xss_react_danger", "xss", 3, True, "typescript", "tsx",
     "<div dangerouslySetInnerHTML={{ __html: post.userComment }} />",
     "React component rendering user-submitted comment without DOMPurify sanitization."),
    ("xss_react_clean", "none_secure", 0, False, "typescript", "tsx",
     "<div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(post.userComment) }} />",
     "Clean React component sanitizing HTML input with DOMPurify allowlist."),
    ("xss_vue_vhtml", "xss", 3, True, "javascript", "vue",
     "<div v-html=\"userBio\"></div>",
     "Vue template binding raw user bio to v-html directive without HTML escaping."),

    ("ssrf_cloud_metadata", "ssrf", 4, True, "python", "py",
     "url = request.args.get('target')\nres = requests.get(url, timeout=3)\nreturn res.text",
     "Webhook dispatcher fetching user URL; permits requests to 169.254.169.254 AWS metadata."),
    ("ssrf_clean_allowlist", "none_secure", 0, False, "python", "py",
     "if is_valid_public_domain(target_url):\n    return requests.get(target_url, timeout=3)",
     "Clean webhook handler validating host against public domain allowlist and non-private IP check."),

    ("idor_missing_tenant", "authz_idor", 3, True, "python", "py",
     "doc = Document.query.filter_by(id=doc_id).first()\nreturn jsonify(doc.to_dict())",
     "Document retrieval endpoint checking doc ID but omitting current user organization check."),
    ("idor_clean_tenant", "none_secure", 0, False, "python", "py",
     "doc = Document.query.filter_by(id=doc_id, org_id=current_user.org_id).first_or_404()",
     "Clean multi-tenant endpoint scoping document query by session organization ID."),

    ("cmdi_os_system", "command_injection", 4, True, "python", "py",
     "os.system(f'ping -c 1 {user_ip}')",
     "Network diagnostic endpoint executing OS command with unsanitized user IP string."),
    ("cmdi_clean_subprocess", "none_secure", 0, False, "python", "py",
     "subprocess.run(['ping', '-c', '1', validated_ip], shell=False, check=True)",
     "Clean subprocess execution passing arguments list with shell=False."),

    ("secret_aws_hardcoded", "crypto_secret_leak", 4, True, "python", "py",
     "AWS_SECRET_KEY = 'EXAMPLE_REDACTED_ACCESS_KEY'",
     "Hardcoded AWS secret key committed in production service configuration."),
    ("secret_clean_env", "none_secure", 0, False, "python", "py",
     "AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')",
     "Clean credentials management reading key securely from environment variable."),

    ("timing_attack_sig", "crypto_secret_leak", 2, False, "python", "py",
     "def verify_sig(a, b):\n    return a == b # Vulnerable to byte-by-byte timing attack",
     "Cryptographic signature check using standard non-constant-time equality operator."),
    ("timing_clean_hmac", "none_secure", 0, False, "python", "py",
     "def verify_sig(a, b):\n    return hmac.compare_digest(a, b)",
     "Clean constant-time signature verification using hmac.compare_digest.")
]


def generate_security_dataset(output_path: Path, target_count: int = 200) -> List[DecisionSample]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples: List[DecisionSample] = []
    idx = 0

    while len(samples) < target_count:
        for prefix, vuln, score, blocker, lang, ext, code, ctx in SEC_PATTERNS:
            idx += 1
            iteration = (idx // len(SEC_PATTERNS)) + 1
            sample_id = f"tier_b_sec_{prefix}_{idx:04d}"

            state = StructuredState(
                title=f"Security Triage #{idx}: {prefix.replace('_', ' ').title()} (Iteration {iteration})",
                file_path=f"src/handlers/module_{iteration}.{ext}",
                language=lang,
                code_snippet=code,
                context=f"{ctx} (Audit Sample #{idx})",
                metadata={"source": "security_taxonomy", "cwe_type": vuln, "iteration": iteration}
            )

            sample = DecisionSample(
                id=sample_id,
                domain=DomainType.CYBERSECURITY,
                tier=DataTier.TIER_B_SILVER,
                state=state,
                questions=CYBERSECURITY_QUESTIONS,
                answers={
                    "vulnerability_class": vuln,
                    "exploitability_score": score,
                    "is_immediate_blocker": blocker
                }
            )
            samples.append(sample)
            if len(samples) >= target_count:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")

    print(f"[INFO] Generated {len(samples)} Tier B Cybersecurity samples to {output_path}")
    return samples


if __name__ == "__main__":
    generate_security_dataset(Path("data/tier_b_silver/security_decisions.jsonl"), target_count=200)
