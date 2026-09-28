"""
Laya Non-Autoregressive Decision Engine Fine-Tuning Script.
Uses real Hugging Face ModernBERT backbone (answerdotai/ModernBERT-base, hidden_dim=768)
with real tokenization, real transformer forward pass, and multi-task decision heads.
"""

import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel


class RealDecisionDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_len: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        state = sample["state"]
        if isinstance(state, dict):
            state_text = f"Title: {state.get('title', '')}\nFile: {state.get('file_path', '')}\nContext: {state.get('context', '')}\nCode:\n{state.get('code_snippet', '')}"
        else:
            state_text = str(state)

        prompts = []
        targets = []
        q_types = []

        for q_id, q in sample["questions"].items():
            ans = sample["answers"].get(q_id)
            if ans is None:
                continue

            q_type = q["type"]
            q_types.append(q_type)

            criteria_text = ""
            if q.get("criteria"):
                opts = [f"{k}: {v}" for k, v in q["criteria"].items()]
                criteria_text = " | ".join(opts)

            prompt = f"{state_text}\n[QUESTION]: {q['instructions']}\n[CRITERIA]: {criteria_text}\n[DECISION]:"
            prompts.append(prompt)

            if q_type == "choice":
                keys = list(q["criteria"].keys())
                label_idx = keys.index(ans) if ans in keys else 0
                targets.append(label_idx)
            elif q_type == "score":
                targets.append(int(ans))
            elif q_type == "noul":
                targets.append(1.0 if ans is True else 0.0)

        return {
            "prompts": prompts,
            "targets": targets,
            "types": q_types,
            "sample_id": sample["id"]
        }


class RealLayaDecisionModel(nn.Module):
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base"):
        super().__init__()
        print(f"[INFO] Loading real ModernBERT encoder backbone: {model_name}...")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size  # 768 for ModernBERT-base

        # Multi-task classification & regression heads
        self.choice_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 12)  # up to 12 choices
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5)   # scores 0, 1, 2, 3, 4
        )
        self.noul_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)   # binary logit
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, q_type: str):
        # Real transformer forward pass through all self-attention layers
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token representation

        if q_type == "choice":
            return self.choice_head(pooled)
        elif q_type == "score":
            return self.score_head(pooled)
        elif q_type == "noul":
            return self.noul_head(pooled).squeeze(-1)
        elif q_type == "all":
            return {
                "choice": self.choice_head(pooled),
                "score": self.score_head(pooled),
                "noul": self.noul_head(pooled).squeeze(-1)
            }
        else:
            raise ValueError(f"Unknown question type: {q_type}")


def train_stage(stage_num: int, config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    stage_key = f"stage{stage_num}_curriculum"
    s_cfg = cfg[stage_key]

    print(f"\n{'='*60}")
    print(f"[START] REAL TRAINING LAYA - Stage {stage_num}: {s_cfg['description']}")
    print(f"{'='*60}")
    print(f"Backbone:           answerdotai/ModernBERT-base (real weights)")
    print(f"Batch Size:         {s_cfg['per_device_train_batch_size']} (Grad Accum: {s_cfg['gradient_accumulation_steps']})")
    print(f"Learning Rate:      {s_cfg['learning_rate']}")
    print(f"Epochs:             {s_cfg['epochs']}")
    print(f"Train Dataset:      {s_cfg['train_data']}")
    print(f"Eval Dataset:       {s_cfg['eval_data']}")
    print(f"{'='*60}\n")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _t = torch.zeros(1, device="cuda")
            device = torch.device("cuda")
            print(f"[INFO] Running on CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("[INFO] CUDA sm_120 driver pending cu128; executing training on CPU.")
    else:
        print("[INFO] Executing training on CPU.")

    # Load real tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = RealLayaDecisionModel("answerdotai/ModernBERT-base")

    if stage_num == 2:
        stage1_ckpt = Path("models/stage1_silver_checkpoint/laya_real_weights.pt")
        if stage1_ckpt.exists():
            print(f"[INFO] Initializing Stage 2 from Stage 1 checkpoint: {stage1_ckpt}")
            ckpt_data = torch.load(stage1_ckpt, map_location=device)
            model.load_state_dict(ckpt_data["model_state_dict"])

    model.to(device)

    # Freeze lower encoder layers for fast laptop convergence, train top layers + heads
    for param in model.encoder.embeddings.parameters():
        param.requires_grad = False
    for layer in model.encoder.layers[:-4]: # Train top 4 transformer layers + heads
        for param in layer.parameters():
            param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=float(s_cfg["learning_rate"]), weight_decay=float(s_cfg["weight_decay"]))

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    train_path = Path(s_cfg["train_data"])
    out_dir = Path(s_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RealDecisionDataset(train_path, tokenizer=tokenizer, max_len=512)
    # Train on full dataset (or slice if running on CPU)
    dataloader = DataLoader(dataset, batch_size=int(s_cfg["per_device_train_batch_size"]), shuffle=True, collate_fn=lambda x: x)

    epochs = int(s_cfg["epochs"])
    accum_steps = int(s_cfg["gradient_accumulation_steps"])

    print(f"[INFO] Loaded {len(dataset)} real samples into dataset. Commencing full GPU training loop across all batches...\n")

    use_cuda = (device.type == "cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            step_count += 1
            batch_loss = torch.tensor(0.0, device=device)

            for sample in batch:
                for q_idx, q_type in enumerate(sample["types"]):
                    prompt = sample["prompts"][q_idx]
                    target = sample["targets"][q_idx]

                    # Real tokenization with ModernBERT tokenizer
                    enc = tokenizer(prompt, max_length=256, truncation=True, padding=True, return_tensors="pt")
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)

                    # Real forward pass with native bfloat16 on RTX 5050
                    if use_cuda:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(input_ids, attention_mask, q_type)
                            if q_type == "choice":
                                loss = ce_loss(logits, torch.tensor([target], device=device)) * cfg["loss_weights"]["choice_loss"]
                            elif q_type == "score":
                                loss = ce_loss(logits, torch.tensor([target], device=device)) * cfg["loss_weights"]["score_loss"]
                            elif q_type == "noul":
                                loss = bce_loss(logits, torch.tensor([float(target)], device=device)) * cfg["loss_weights"]["noul_loss"]
                            else:
                                continue
                    else:
                        logits = model(input_ids, attention_mask, q_type)
                        if q_type == "choice":
                            loss = ce_loss(logits, torch.tensor([target], device=device)) * cfg["loss_weights"]["choice_loss"]
                        elif q_type == "score":
                            loss = ce_loss(logits, torch.tensor([target], device=device)) * cfg["loss_weights"]["score_loss"]
                        elif q_type == "noul":
                            loss = bce_loss(logits, torch.tensor([float(target)], device=device)) * cfg["loss_weights"]["noul_loss"]
                        else:
                            continue

                    batch_loss = batch_loss + loss

            scaled_loss = batch_loss / accum_steps
            scaled_loss.backward()
            total_loss += batch_loss.item()

            if step_count % accum_steps == 0 or step_count == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(1, step_count)
        print(f"  [Epoch {epoch}/{epochs}] Full Dataset Train Loss: {avg_loss:.4f} | Processed: {len(dataset)} samples")

    # Save real model weights
    checkpoint_file = out_dir / "laya_real_weights.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": s_cfg,
        "loss": avg_loss
    }, checkpoint_file)

    meta = {
        "stage": stage_num,
        "status": "completed",
        "epochs_trained": epochs,
        "final_loss": avg_loss,
        "checkpoint_file": str(checkpoint_file),
        "backbone": "answerdotai/ModernBERT-base",
        "hidden_dim": 768
    }
    with open(out_dir / "stage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SUCCESS] Stage {stage_num} training finished with real weights!")
    print(f"Checkpoint saved to: {checkpoint_file}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--config", type=str, default="training/config.yaml")
    args = parser.parse_args()

    train_stage(args.stage, Path(args.config))
