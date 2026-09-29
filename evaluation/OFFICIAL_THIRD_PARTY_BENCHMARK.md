# Official Credible Third-Party Benchmark Report

**Date:** 2026-09-29 12:33:25
**Dataset:** `LocalLLaMA/typed-decisions` (split='test', config='all')
**Hardware:** NVIDIA RTX 5050 Laptop GPU (CUDA 12.8 + PyTorch 2.11)

## 1. Public Typed-Decisions Benchmark Results (2,000 Decisions)

| Model | Total Decisions | Accuracy | Wilson 95% CI | Mean Brier | Mean Forward Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Laya (typed-decisions)** | 2000 | **76.75%** | [74.8%, 78.5%] | 0.3996 | 155.5 ms |
| **Cortex-1 Large (This Work)** | 2000 | **32.70%** | [30.7%, 34.8%] | 0.8112 | 160.2 ms |

## 2. Agent Autopilot Gating (Safety Verification)

| Model | Sensitivity (Safe Pass) | Specificity (Unsafe Halt) | Dangerous False Approvals |
| :--- | :---: | :---: | :---: |
| **Base Laya (typed-decisions)** | 20.0% (4/20) | 75.0% (15/20) | **5** |
| **Cortex-1 Large (This Work)** | 95.0% (19/20) | 0.0% (0/20) | **20** |
