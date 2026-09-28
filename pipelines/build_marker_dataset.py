"""
Preprocess JSONL samples into marker-indexed token sequences for ModernBERT-large.
Uses laya.common.build_sequence and render_options.
Saves pre-tokenized datasets to:
- data/marker_train.pt
- data/marker_val.pt
"""

import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer
from laya.common import build_sequence, render_options, QTYPES


def build_marker_item(sample: Dict[str, Any], q_id: str, q: Dict[str, Any], tok, max_len: int = 512, head_max_len: int = 192):
    t = q["type"]
    crit = q.get("criteria")
    q_dict = {"t": t, "ins": q["instructions"], "crit": crit}
    opts = render_options(q_dict)
    k = len(opts)

    ans = sample["answers"].get(q_id)
    if ans is None:
        return None

    # Target probability construction
    if t == "noul":
        target = [1.0, 0.0] if ans is False else [0.0, 1.0]
        label = 0 if ans is False else 1
    elif t == "score":
        target = [0.0] * k
        score_val = int(ans)
        if 0 <= score_val < k:
            target[score_val] = 1.0
            label = score_val
        else:
            target[0] = 1.0
            label = 0
    elif t == "choice":
        keys = list(crit.keys()) if isinstance(crit, dict) else []
        target = [0.0] * k
        if ans in keys:
            idx = keys.index(ans)
            target[idx] = 1.0
            label = idx
        else:
            target[0] = 1.0
            label = 0
    else:
        return None

    seq, markers = build_sequence(tok, sample["state"], q_dict, max_len=max_len, head_max_len=head_max_len)
    if len(markers) != k:
        return None

    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES[t],
        "target": target,
        "label": label,
        "sample_id": sample.get("id", ""),
        "q_id": q_id,
        "domain": sample.get("domain", ""),
        "is_keyword_free": sample.get("state", {}).get("metadata", {}).get("is_keyword_free", False)
    }


def process_dataset(jsonl_path: Path, output_pt: Path, tok, max_len: int = 512):
    print(f"\n[PREPROCESSING] Tokenizing {jsonl_path} -> {output_pt}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    items = []
    skipped = 0

    for s in samples:
        for q_id, q in s["questions"].items():
            item = build_marker_item(s, q_id, q, tok, max_len=max_len)
            if item is not None:
                items.append(item)
            else:
                skipped += 1

    print(f"Processed {len(samples)} cases into {len(items)} decision sequences (Skipped: {skipped}).")
    torch.save(items, output_pt)
    print(f"[SAVED] {output_pt} ({len(items)} items).")
    return items


def main():
    print(f"{'='*60}")
    print("LAYA REFERENCE DATASET TOKENIZATION (ModernBERT-large)")
    print(f"{'='*60}")

    model_id = "answerdotai/ModernBERT-large"
    print(f"Loading tokenizer: {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)

    # Support upgraded high-quality datasets by default
    upgraded_train = Path("data/upgraded_train.jsonl")
    upgraded_val = Path("data/upgraded_val.jsonl")

    if upgraded_train.exists():
        print("[INFO] Upgraded high-grade datasets found! Tokenizing upgraded datasets...")
        train_jsonl = upgraded_train
        val_jsonl = upgraded_val
        out_train_pt = Path("data/marker_train_upgraded.pt")
        out_val_pt = Path("data/marker_val_upgraded.pt")
    else:
        train_jsonl = Path("data/stage2_train_rebalanced.jsonl")
        val_jsonl = Path("data/stage2_val.jsonl")
        out_train_pt = Path("data/marker_train.pt")
        out_val_pt = Path("data/marker_val.pt")

    train_items = process_dataset(train_jsonl, out_train_pt, tok, max_len=512)
    val_items = process_dataset(val_jsonl, out_val_pt, tok, max_len=512)

    # Also link/copy to marker_train.pt and marker_val.pt for seamless trainer compatibility
    torch.save(train_items, Path("data/marker_train.pt"))
    torch.save(val_items, Path("data/marker_val.pt"))
    print("[SAVED] Synchronized active training sets: data/marker_train.pt & data/marker_val.pt")

    # Verification checks
    noul_train = [it for it in train_items if it["qtype"] == QTYPES["noul"]]
    pos_train = sum(1 for it in noul_train if it["label"] == 1)
    neg_train = sum(1 for it in noul_train if it["label"] == 0)

    noul_val = [it for it in val_items if it["qtype"] == QTYPES["noul"]]
    pos_val = sum(1 for it in noul_val if it["label"] == 1)
    neg_val = sum(1 for it in noul_val if it["label"] == 0)

    print("\n[VERIFICATION OF TOKENIZED DATA]")
    print(f"Total Train Sequences: {len(train_items)}")
    print(f"Total Val Sequences:   {len(val_items)}")
    print(f"Train NOUL Balance:    True={pos_train}, False={neg_train} (Total={len(noul_train)})")
    print(f"Val NOUL Balance:      True={pos_val}, False={neg_val} (Total={len(noul_val)})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
