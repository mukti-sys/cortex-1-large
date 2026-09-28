import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
from training.train_laya import RealLayaDecisionModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
model = RealLayaDecisionModel('answerdotai/ModernBERT-base')
ckpt = torch.load('models/laya_final_brain/laya_real_weights.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

for split_name, path in [('Set 1', 'data/eval/set1_generic.jsonl'), ('Set 2', 'data/eval/set2_personal.jsonl')]:
    with open(path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    preds = []
    truths = []
    probs = []
    with torch.no_grad():
        for s in samples:
            state = s['state']
            if isinstance(state, dict):
                state_text = f"Title: {state.get('title', '')}\nContext: {state.get('context', '')}\nCode:\n{state.get('code_snippet', '')}"
            else:
                state_text = str(state)
            for q_id, q in s['questions'].items():
                if q['type'] == 'noul':
                    criteria_text = ""
                    if q.get('criteria'):
                        criteria_text = " | ".join([f"{k}: {v}" for k, v in q['criteria'].items()])
                    prompt = f"{state_text}\n[QUESTION]: {q['instructions']}\n[CRITERIA]: {criteria_text}\n[DECISION]:"
                    enc = tokenizer(prompt, max_length=256, truncation=True, padding=True, return_tensors='pt').to(device)
                    logit = model(enc['input_ids'], enc['attention_mask'], 'noul').item()
                    prob = torch.sigmoid(torch.tensor(logit)).item()
                    pred = prob >= 0.5
                    preds.append(pred)
                    truths.append(s['answers'][q_id])
                    probs.append(prob)
    num_pred_true = sum(1 for p in preds if p is True)
    num_pred_false = sum(1 for p in preds if p is False)
    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    print(f"{split_name}: Total={len(preds)}, Pred True={num_pred_true}, Pred False={num_pred_false}, Correct={correct}/{len(preds)} ({correct/len(preds)*100:.1f}%), Avg Prob={sum(probs)/len(probs):.4f}")
