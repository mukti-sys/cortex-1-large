import sys
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 60)
print("NVIDIA RTX 5050 CUDA VERIFICATION")
print("=" * 60)
print("PyTorch Version:    ", torch.__version__)
print("CUDA Build:         ", torch.version.cuda)
print("CUDA Available:     ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device Name:        ", torch.cuda.get_device_name(0))
    print("Compute Capability: ", torch.cuda.get_device_capability(0))
    print("Total VRAM (GB):    ", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
    print("-" * 60)
    try:
        device = torch.device("cuda:0")
        # Test real bfloat16 matrix multiplication on Blackwell architecture
        a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print(f"[SUCCESS] Real bfloat16 tensor matmul (1024x1024) succeeded on RTX 5050!")
        print(f"Matrix Sum Check:   {c.sum().item():.4f}")
    except Exception as e:
        print(f"[FAILED] CUDA Execution Error: {e}")
else:
    print("[WARN] CUDA is not available in this environment.")
print("=" * 60)
