"""
Antigravity Artifact & Conversation Extractor (Tier A - Gold).
Mines local Antigravity transcripts and implementation plans from ~/.gemini/antigravity-ide/brain/
to extract real user approvals, plan structures, and tool calls into Laya Gold-tier samples.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from schemas.primitives import DecisionSample, StructuredState, DomainType, DataTier
from schemas.questions_catalog import AUTOPILOT_GATING_QUESTIONS


def extract_from_brain_artifacts(brain_root: Path, output_file: Path) -> List[DecisionSample]:
    samples: List[DecisionSample] = []
    if not brain_root.exists():
        print(f"Brain directory not found at: {brain_root}")
        return samples

    # Scan all conversation folders in brain_root
    conv_dirs = [d for d in brain_root.iterdir() if d.is_dir()]
    print(f"Found {len(conv_dirs)} conversation folders in {brain_root}")

    sample_idx = 0
    for conv_dir in conv_dirs:
        # Check for implementation_plan.md
        plan_path = conv_dir / "implementation_plan.md"
        if plan_path.exists():
            try:
                content = plan_path.read_text(encoding="utf-8", errors="ignore")
                
                # Check plan length and characteristics
                has_breaking_warning = "CRITICAL" in content or "BREAKING" in content or "WARNING" in content
                risk = 3 if has_breaking_warning else (1 if len(content) < 1500 else 2)
                autopilot = not has_breaking_warning and risk <= 1
                route = "deep_reasoning" if risk >= 3 else ("balanced_agent" if risk >= 1 else "fast_system1")

                state = StructuredState(
                    title=f"Plan from {conv_dir.name[:8]}",
                    context=content[:2000],  # keep within context window
                    metadata={"conv_id": conv_dir.name, "source": "implementation_plan.md"}
                )

                sample = DecisionSample(
                    id=f"tier_a_plan_{sample_idx:04d}",
                    domain=DomainType.AGENT_AUTOPILOT,
                    tier=DataTier.TIER_A_GOLD,
                    state=state,
                    questions=AUTOPILOT_GATING_QUESTIONS,
                    answers={
                        "should_autopilot": autopilot,
                        "risk_score": risk,
                        "route_task": route
                    }
                )
                samples.append(sample)
                sample_idx += 1
            except Exception as e:
                print(f"Error parsing plan in {conv_dir}: {e}")

        # Check for transcript logs to extract tool actions
        logs_dir = conv_dir / ".system_generated" / "logs"
        transcript_file = logs_dir / "transcript.jsonl"
        if transcript_file.exists():
            try:
                with open(transcript_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if '"type":"USER_INPUT"' in line:
                            data = json.loads(line)
                            user_prompt = data.get("content", "")
                            if len(user_prompt) > 30:
                                # Estimate task routing and risk from user prompt
                                is_deep = any(w in user_prompt.lower() for w in ["refactor", "architect", "security", "pipeline", "benchmark", "fine tune"])
                                is_fast = any(w in user_prompt.lower() for w in ["typo", "format", "rename", "lint", "comment"])
                                
                                route = "deep_reasoning" if is_deep else ("fast_system1" if is_fast else "balanced_agent")
                                risk = 2 if is_deep else (0 if is_fast else 1)
                                autopilot = not is_deep and risk <= 1

                                sample = DecisionSample(
                                    id=f"tier_a_prompt_{sample_idx:04d}",
                                    domain=DomainType.AGENT_AUTOPILOT,
                                    tier=DataTier.TIER_A_GOLD,
                                    state=StructuredState(
                                        title="User Task Prompt",
                                        context=user_prompt[:1500],
                                        metadata={"conv_id": conv_dir.name}
                                    ),
                                    questions=AUTOPILOT_GATING_QUESTIONS,
                                    answers={
                                        "should_autopilot": autopilot,
                                        "risk_score": risk,
                                        "route_task": route
                                    }
                                )
                                samples.append(sample)
                                sample_idx += 1
                                if sample_idx >= 500:
                                    break
            except Exception as e:
                pass

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")

    print(f"Extracted {len(samples)} Tier A (Gold) samples saved to {output_file}")
    return samples


if __name__ == "__main__":
    app_data_brain = Path(os.path.expanduser(r"~\.gemini\antigravity-ide\brain"))
    out_path = Path("data/tier_a_gold/antigravity_history.jsonl")
    extract_from_brain_artifacts(app_data_brain, out_path)
