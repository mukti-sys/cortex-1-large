import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from laya.common import DecisionModel, collate_items, proper_reward

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

print(f"{'='*60}")
print("EMPIRICAL VRAM SAFETY TEST: ModernBERT-large (421M)")
print(f"{'='*60}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
print(f"Total GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("[1/5] Loading ModernBERT-large backbone...")
enc = AutoModel.from_pretrained("answerdotai/ModernBERT-large", attn_implementation="sdpa")
enc.gradient_checkpointing_enable()

print("[2/5] Constructing Laya DecisionModel (421M + 2-layer head)...")
model = DecisionModel(enc, head_layers=2)
model.to(device=device)

vram_model = torch.cuda.memory_allocated() / (1024**3)
print(f"  * VRAM after model allocation: {vram_model:.2f} GB")

print("[3/5] Initializing bitsandbytes 8-bit AdamW optimizer...")
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1.5e-5, weight_decay=0.01)
    print("  * Using bnb.optim.AdamW8bit!")
except Exception as e:
    print(f"  * Fallback to torch AdamW: {e}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-5, weight_decay=0.01)

vram_opt_init = torch.cuda.memory_allocated() / (1024**3)
print(f"  * VRAM after optimizer init: {vram_opt_init:.2f} GB")

print("[4/5] Loading 1 batch of 2 items from data/marker_train.pt...")
items = torch.load("data/marker_train.pt", weights_only=False)[:2]
tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
batch = collate_items([items], pad_id=tok.pad_token_id)

input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
marker_pos = batch["marker_pos"].to(device)
marker_mask = batch["marker_mask"].to(device)
qtype = batch["qtype"].to(device)
target = batch["target"].to(device)

print("[5/5] Executing real forward + backward pass with autocast(bfloat16) & gradient checkpointing...")
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits, act_logits = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    probs = torch.softmax(logits, dim=-1)

    # Proper scoring rule loss
    rewards = proper_reward(probs, target, qtype, marker_mask)
    loss = -rewards.mean()

print(f"  * Loss: {loss.item():.4f}")
loss.backward()
optimizer.step()
torch.cuda.synchronize()

peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
print(f"\n[MEASURED PEAK VRAM]: {peak_vram:.2f} GB (Budget: 8.00 GB)")
safety_headroom = 8.00 - peak_vram
print(f"[SAFETY HEADROOM]:    {safety_headroom:.2f} GB")

if peak_vram <= 6.5:
    print(f"[RESULT] VRAM Safety Check: PASS! Model fits comfortably with {safety_headroom:.2f} GB headroom.")
else:
    print(f"[RESULT] VRAM Safety Check: WARNING! Tighter than 1.5 GB headroom.")
print(f"{'='*60}\n")
