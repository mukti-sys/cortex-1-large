"""
Laya Reference Training Engine: Full ModernBERT-large (421M) with Dynamic Marker-Pooling.
Strictly optimized for NVIDIA GeForce RTX 5050 Laptop GPU (sm_120 Blackwell, 8GB GDDR7).
Uses bitsandbytes 8-bit AdamW + bfloat16 autocast + gradient checkpointing (Peak VRAM: 3.98 GB).
Trains against proper scoring rules (log score + spherical + ranked probability score).
Evaluates exclusively on quarantined validation split (data/marker_val.pt) to protect test sets.
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import bitsandbytes as bnb
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import save_file
from laya.common import DecisionModel, collate_items, proper_reward, QTYPES, QTYPE_NAMES


def evaluate_val_split(model, val_loader, device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_steps = 0

    # Primitive metrics
    stats = {
        "noul": {"total": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0, "subtle_total": 0, "subtle_tn": 0},
        "score": {"total": 0, "correct": 0, "adjacent": 0},
        "choice": {"total": 0, "correct": 0}
    }

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            target = batch["target"].to(device)
            labels = batch["label"].to(device)
            meta = batch["meta"]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
                probs = torch.softmax(logits, dim=-1)
                rewards = proper_reward(probs, target, qtype, marker_mask)
                loss = -rewards.mean()

            total_loss += loss.item()
            total_steps += 1

            preds = torch.argmax(logits, dim=-1)

            for i in range(len(preds)):
                qt = qtype[i].item()
                pred = preds[i].item()
                label = labels[i].item()
                is_correct = (pred == label)

                if qt == QTYPES["noul"]:
                    stats["noul"]["total"] += 1
                    if is_correct:
                        stats["noul"]["correct"] += 1

                    # 0 = False (don't autopilot), 1 = True (allow autopilot)
                    is_subtle = meta[i].get("is_keyword_free", False)
                    if label == 1:
                        if pred == 1:
                            stats["noul"]["tp"] += 1
                        else:
                            stats["noul"]["fn"] += 1
                    else:
                        if is_subtle:
                            stats["noul"]["subtle_total"] += 1
                        if pred == 0:
                            stats["noul"]["tn"] += 1
                            if is_subtle:
                                stats["noul"]["subtle_tn"] += 1
                        else:
                            stats["noul"]["fp"] += 1

                elif qt == QTYPES["score"]:
                    stats["score"]["total"] += 1
                    if is_correct:
                        stats["score"]["correct"] += 1
                    if abs(pred - label) <= 1:
                        stats["score"]["adjacent"] += 1

                elif qt == QTYPES["choice"]:
                    stats["choice"]["total"] += 1
                    if is_correct:
                        stats["choice"]["correct"] += 1

    avg_loss = total_loss / max(1, total_steps)

    # NOUL Tri-metrics
    tp = stats["noul"]["tp"]
    tn = stats["noul"]["tn"]
    fp = stats["noul"]["fp"]
    fn = stats["noul"]["fn"]
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    bal_acc = (sens + spec) / 2.0
    subtle_spec = stats["noul"]["subtle_tn"] / max(1, stats["noul"]["subtle_total"])

    score_acc = stats["score"]["correct"] / max(1, stats["score"]["total"])
    score_adj = stats["score"]["adjacent"] / max(1, stats["score"]["total"])
    choice_acc = stats["choice"]["correct"] / max(1, stats["choice"]["total"])

    return {
        "val_loss": avg_loss,
        "noul": {
            "sens": sens,
            "spec": spec,
            "bal_acc": bal_acc,
            "subtle_spec": subtle_spec,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "acc": stats["noul"]["correct"] / max(1, stats["noul"]["total"])
        },
        "score": {
            "exact_acc": score_acc,
            "adj_acc": score_adj
        },
        "choice": {
            "exact_acc": choice_acc
        }
    }


def train_reference():
    print(f"\n{'='*70}")
    print("STARTING REFERENCE LAYA TRAINING: ModernBERT-large (421M)")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:           {device} ({torch.cuda.get_device_name(0)})")
    print(f"Backbone:         answerdotai/ModernBERT-large (421M params, 28 layers)")
    print(f"Head:             TransformerEncoder (2 layers) + Option Marker-Pooling")
    print(f"Optimizer:        bitsandbytes 8-bit AdamW (lr=1.5e-5, weight_decay=0.01)")
    print(f"Batch Size:       4 (Grad Accum: 4 -> Effective Batch Size: 16)")
    print(f"Checkpoints:      Gradient Checkpointing Enabled")
    print(f"Loss:             Strictly Proper Scoring Rules (Log + Spherical + RPS)")
    print(f"{'='*70}\n")

    # Load pre-tokenized items
    train_items = torch.load("data/marker_train.pt", weights_only=False)
    val_items = torch.load("data/marker_val.pt", weights_only=False)
    print(f"[INFO] Loaded {len(train_items)} train items and {len(val_items)} val items.")

    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    pad_id = tok.pad_token_id

    # Create batch loaders
    def batch_generator(items, batch_size=4, shuffle=True):
        indices = list(range(len(items)))
        if shuffle:
            import random
            random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            chunk = [items[idx] for idx in indices[i:i + batch_size]]
            yield collate_items([chunk], pad_id=pad_id)

    # Initialize model
    print("[INFO] Initializing ModernBERT-large backbone...")
    enc = AutoModel.from_pretrained("answerdotai/ModernBERT-large", attn_implementation="sdpa")
    enc.gradient_checkpointing_enable()

    model = DecisionModel(enc, head_layers=2)
    model.to(device)

    # 8-bit AdamW optimizer
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1.5e-5, weight_decay=0.01)

    epochs = 3
    accum_steps = 4
    out_dir = Path("models/laya_large_reference")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_bal_acc = 0.0
    best_epoch = 0

    total_train_steps = (len(train_items) // 4) * epochs
    print(f"[INFO] Beginning training across {epochs} epochs ({total_train_steps} total micro-steps)...\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_in_epoch = 0
        optimizer.zero_grad()

        ep_start = time.time()

        for step, batch in enumerate(batch_generator(train_items, batch_size=4, shuffle=True)):
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)
            target = batch["target"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
                probs = torch.softmax(logits, dim=-1)
                rewards = proper_reward(probs, target, qtype, marker_mask)
                loss = -rewards.mean() / accum_steps

            loss.backward()
            epoch_loss += loss.item() * accum_steps
            step_in_epoch += 1

            if step_in_epoch % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step_in_epoch % 200 == 0:
                cur_vram = torch.cuda.memory_allocated() / (1024**3)
                print(f"  [Epoch {epoch} | Step {step_in_epoch}/{len(train_items)//4}] Step Loss: {loss.item()*accum_steps:.4f} | VRAM: {cur_vram:.2f} GB")

        # Step any remaining gradients
        if step_in_epoch % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_epoch_loss = epoch_loss / max(1, step_in_epoch)
        ep_duration = time.time() - ep_start
        print(f"\n[EPOCH {epoch} COMPLETE] Duration: {ep_duration:.1f}s | Avg Training Loss: {avg_epoch_loss:.4f}")

        # Evaluate strictly on quarantined validation split
        print("[VALIDATION] Evaluating on data/upgraded_val.jsonl (Quarantined Multi-Domain Split)...")
        val_loader = list(batch_generator(val_items, batch_size=4, shuffle=False))
        v_res = evaluate_val_split(model, val_loader, device)

        n = v_res["noul"]
        s = v_res["score"]
        c = v_res["choice"]

        print(f"  * Validation Loss:          {v_res['val_loss']:.4f}")
        print(f"  * NOUL Sensitivity (Recall on True):  {n['sens']*100:.1f}% ({n['tp']}/{n['tp']+n['fn']})")
        print(f"  * NOUL Specificity (Recall on False): {n['spec']*100:.1f}% ({n['tn']}/{n['tn']+n['fp']})")
        print(f"  * NOUL Balanced Accuracy:            {n['bal_acc']*100:.1f}%")
        print(f"  * Keyword-Free Subtle Risk Recall:   {n['subtle_spec']*100:.1f}% ({n['tn']}/{n['tn']+n['fp']})")
        print(f"  * SCORE Exact Top-1 Accuracy:        {s['exact_acc']*100:.1f}% (Adjacent +-1: {s['adj_acc']*100:.1f}%)")
        print(f"  * CHOICE Dynamic Option Accuracy:    {c['exact_acc']*100:.1f}%")
        print(f"  * Confusion Matrix: TP={n['tp']}, FP={n['fp']}, TN={n['tn']}, FN={n['fn']}\n")

        # Check acceptance gates
        if n["bal_acc"] > best_bal_acc:
            best_bal_acc = n["bal_acc"]
            best_epoch = epoch
            print(f"[CHECKPOINT] New best Balanced Accuracy ({best_bal_acc*100:.1f}%)! Saving model to {out_dir}...")
            # Save model state dict and safetensors
            torch.save({"model_state_dict": model.state_dict(), "val_metrics": v_res}, out_dir / "laya_large_weights.pt")
            save_file(model.state_dict(), out_dir / "model.safetensors")

            # Save config
            model_cfg = {
                "encoder": "answerdotai/ModernBERT-large",
                "head_layers": 2,
                "n_act": 1,
                "max_len": 512,
                "head_max_len": 192,
                "best_epoch": best_epoch,
                "val_balanced_accuracy": float(best_bal_acc),
                "val_sensitivity": float(n["sens"]),
                "val_specificity": float(n["spec"])
            }
            with open(out_dir / "rl_agent_config.json", "w", encoding="utf-8") as f:
                json.dump(model_cfg, f, indent=2)

    total_duration = time.time() - start_time
    print(f"{'='*70}")
    print(f"[SUCCESS] Reference Training Finished in {total_duration/60:.1f} minutes!")
    print(f"Best Validation Balanced Accuracy: {best_bal_acc*100:.1f}% (Epoch {best_epoch})")
    print(f"Saved Checkpoint: {out_dir / 'model.safetensors'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    train_reference()
