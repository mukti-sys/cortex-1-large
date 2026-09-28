import json
import torch
import laya

print("Testing official Laya loader on GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

try:
    agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions", device=device)
    print(f"[SUCCESS] Loaded convaiinnovations/laya (typed-decisions) on {device}!")

    with open("data/eval/set1_generic.jsonl", "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    print(f"\nSample ID: {sample['id']}")
    print(f"State: {sample['state']}")
    print(f"Questions: {list(sample['questions'].keys())}")
    
    res = agent.predict(sample["state"], sample["questions"])
    print("\nLaya System 1 Predictions:")
    print(json.dumps(res, indent=2))
    print("\nGround Truth Answers:")
    print(json.dumps(sample["answers"], indent=2))

except Exception as e:
    print(f"[ERROR] {e}")
