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
* **100% Specificity on Safety Stops:** Zero false autonomous approvals across tested multi-file maintainer pull requests.
* **Beats TypeSafe Jev:** Achieves **82.32% Top-1 Accuracy** on held-out industry benchmarks (+9.62% margin over TypeSafe Jev's published 72.70%).

---

## Model Details

* **Backbone:** `answerdotai/ModernBERT-large` (28 layers, 1024 hidden dimension, 421M parameters).
* **Head Architecture:** 2-layer `TransformerEncoder` + Dynamic Marker-Token Pooling (`torch.gather`).
* **Training Precision:** `bfloat16` mixed precision + 8-bit AdamW optimizer with gradient checkpointing.
* **Inference Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB GDDR7, Blackwell `sm_120`).

---

## Benchmarks & Evaluation

### Set 1: Generic Industry Benchmark vs. Competitors (Held-Out)
* **Overall Top-1 Accuracy:** **82.32%** (Wilson 95% CI: $[79.30\%, 84.98\%]$)
* **TypeSafe Jev (Published):** **72.70%** (Margin: **+9.62%**, $p < 0.001$)
* **AI / ML Runtime Triage:** **99.4% (164 / 165 correct)**
* **Cybersecurity CWE Triage:** **95.2% (300 / 315 correct)**

### Set 2: Developer PR Autopilot Gating (Held-Out)
* **Top-1 Accuracy:** **93.66% (399 / 426 correct)**
* **Sensitivity (Safe Actions):** **100.0% (71 / 71)**
* **Specificity (Dangerous Halts):** **100.0% (71 / 71)** — Zero False Approvals (`FP = 0`).

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
