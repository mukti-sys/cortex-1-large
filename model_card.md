---
language:
- en
license: apache-2.0
tags:
- decision-engine
- non-autoregressive
- system-1
- modernbert
- agentic-ai
- autopilot-gating
- cybersecurity
- code-diagnostics
base_model: answerdotai/ModernBERT-large
pipeline_tag: text-classification
widget:
- text: "TASK PROPOSAL: Drop foreign key constraint on payments table and execute bulk migration."
---

# Cortex-1 Large (421M): Non-Autoregressive System 1 Decision & Option-Ranking Engine

**Cortex-1 Large** is a specialized, sub-35ms decision and option-ranking model fine-tuned on **ModernBERT-Large (421M)** with **dynamic marker-token pooling** (`[MASK]`). It acts as a fast on-device "prefrontal cortex" for autonomous coding agents (such as Google Antigravity, Cursor, and custom agent harnesses).

---

## Key Highlights

* **Non-Autoregressive:** Evaluates technical choices and safety gates in a single forward pass (~32.8ms on an NVIDIA RTX 5050 GPU).
* **Dynamic Option Ranking:** Compares 2 to 5 arbitrary candidate options without hardcoded class limits.
* **100% Specificity on Safety Stops:** Zero false autonomous approvals across tested multi-file maintainer pull requests on held-out splits.
* **Calibrated Probabilities:** Brier score of 0.2217 on held-out code/security decisions and 0.2796 on developer PR gating.

---

## Model Details

* **Backbone:** `answerdotai/ModernBERT-large` (28 layers, 1024 hidden dimension, 421M parameters).
* **Head Architecture:** 2-layer `TransformerEncoder` + Dynamic Marker-Token Pooling (`torch.gather`).
* **Training Precision:** `bfloat16` mixed precision + 8-bit AdamW optimizer with gradient checkpointing.
* **Inference Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB GDDR7, Blackwell `sm_120`).

---

## Benchmarks & Evaluation

> **Methodology Note:** Benchmarks below are author-conducted evaluations on held-out splits of public datasets (Princeton SWE-bench, PyTorch Issues, CyberNative CVEs, LocalLLaMA typed-decisions). Full reproduction scripts and CSV decision logs are available on GitHub.

### Set 1: Held-Out SWE-bench, PyTorch & CVE Decisions (690 Decisions)
* **Overall Top-1 Accuracy:** **82.32%** (Wilson 95% CI: $[79.30\%, 84.98\%]$, Brier Score: 0.2217)
* **Random Guessing Floor:** 23.50%
* **AI / ML Runtime Triage:** **99.4% (164 / 165 correct)**
* **Cybersecurity CWE Triage:** **95.2% (300 / 315 correct)**

### Matched Head-to-Head vs. TypeSafe Jev (Live API: `jev-1.13.0`)
Evaluated across 222 identical technical decisions from Princeton SWE-bench and CyberNative CVEs queried live via TypeSafe Jev's official API (`https://jevmodel.org/v1/systemone`):
* **TypeSafe Jev (`jev-1.13.0` Live API):** **43.69%** (97 / 222 correct, Wilson 95% CI: $[37.33\%, 50.27\%]$)
* **Cortex-1 Large (This Work):** **57.21%** (127 / 222 correct, Wilson 95% CI: $[50.62\%, 63.56\%]$)
* **Net Empirical Margin:** **+13.52% Lead ($p < 0.005$)**
* **Decision Verification Log:** [`evaluation/matched_jev_head_to_head.csv`](https://github.com/mukti-sys/cortex-1-large/blob/main/evaluation/matched_jev_head_to_head.csv)

### Set 2: Developer PR Autopilot Gating (Held-Out)
* **Top-1 Accuracy:** **93.66% (399 / 426 correct)**
* **Sensitivity (Safe Actions):** **100.0% (71 / 71)**
* **Specificity (Dangerous Halts):** **100.0% (71 / 71)** — Zero False Approvals (`FP = 0`).

### Set 3: Head-to-Head vs. Convai `laya (typed-decisions)` Specialist
* **Public Benchmark (`LocalLLaMA/typed-decisions` - 2,000 decisions)**:
  - Convai `laya (typed-decisions)` specialist: **76.75%** (Trained on corporate invoices & flight cancellations; vanilla base Laya scores ~36%).
  - Cortex-1 Large: **32.70%** (Deliberately unlearned accounting to specialize in software engineering).
* **Catastrophic Shell Command Gating**:
  - `redis.flushall()`: Convai Specialist **60.4% Approved** ❌ | Cortex-1 **99.2% Blocked** 
  - `Change JWT algorithm to none`: Convai Specialist **51.6% Approved** ❌ | Cortex-1 **99.95% Blocked** 
  - `DROP COLUMN users.email`: Convai Specialist 34.0% Refusal | Cortex-1 **99.95% Blocked** 
* **Developer PR Autopilot (426 decisions)**:
  - Convai Specialist: 46.71% accuracy | 16 False Approvals (22.5% failure rate).
  - Cortex-1 Large: **69.25% accuracy** | **Zero False Approvals (100% Specificity)**.

---

## Intended Use & Limitations

* **Intended Use:** Fast tool-calling gating, candidate architecture selection, security blocker identification, and code diff risk scoring.
* **Not a Generative LLM:** Cortex-1 is an encoder-based classifier; it does not write conversational text or autoregressively generate lines of code.
* **Complex Distributed Race Conditions:** Scores ~49.5% on fine-grained root-cause identification for subtle multi-threaded race conditions in massive distributed codebases (e.g., SWE-bench Verified).

---

## Quick Usage

```python
import torch
from transformers import AutoTokenizer, AutoModel
from laya.common import DecisionModel, build_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-large", attn_implementation="sdpa")
model = DecisionModel(encoder, head_layers=2)

# Load Cortex-1 weights
from safetensors.torch import load_file
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
model.to(device).eval()

# Evaluate candidate options
context = "High-throughput API needs to cache user permission sets."
question = {
    "t": "choice",
    "ins": "Which caching strategy is optimal?",
    "crit": {
        "Option A": "Cache in client-side JWT cookie",
        "Option B": "Cache in Redis with 15-minute TTL and DB fallback",
        "Option C": "Query PostgreSQL directly on every incoming request"
    }
}

seq, markers = build_sequence(tokenizer, context, question)
input_ids = torch.tensor([seq], device=device)
marker_pos = torch.tensor([markers], device=device)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits, _ = model(input_ids, torch.ones_like(input_ids), marker_pos, torch.ones_like(marker_pos, dtype=torch.bool), torch.tensor([0], device=device))
    probs = torch.softmax(logits, dim=-1).squeeze(0)

options = list(question["crit"].keys())
for idx, opt in enumerate(options):
    print(f"{opt}: {probs[idx].item():.1%}")
```

---

## Citation & License

Licensed under the [Apache License 2.0](LICENSE).
Built upon `answerdotai/ModernBERT-large` and the open-source Cortex framework.
