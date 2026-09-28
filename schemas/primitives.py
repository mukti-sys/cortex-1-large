"""
Pydantic primitives and type definitions for Laya Decision Engine.
Guarantees 100% schema consistency across all dataset tiers and pipeline scripts.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


class QuestionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


class DomainType(str, Enum):
    AGENT_AUTOPILOT = "agent_autopilot"
    CYBERSECURITY = "cybersecurity"
    FULLSTACK_WEB = "fullstack_web"
    AI_ML_ENGINEERING = "ai_ml_engineering"
    DEVOPS_CLOUD = "devops_cloud"
    SYSTEMS_PROGRAMMING = "systems_programming"


class DataTier(str, Enum):
    TIER_A_GOLD = "tier_a_gold"        # Personal Antigravity history
    TIER_B_SILVER = "tier_b_silver"    # Public technical datasets (ADR, CVE, SWE-bench)
    TIER_C_SYNTHETIC = "tier_c_synthetic"  # Paraphrased / targeted synthetic examples


class TypedQuestion(BaseModel):
    type: QuestionType
    instructions: str
    criteria: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def validate_criteria(self) -> "TypedQuestion":
        if self.type in (QuestionType.CHOICE, QuestionType.SCORE):
            if not self.criteria or len(self.criteria) == 0:
                raise ValueError(f"Question of type '{self.type.value}' must provide a 'criteria' dictionary.")
        if self.type == QuestionType.SCORE:
            # Check that score criteria keys are numeric strings (e.g. "0", "1", "2", "3", "4")
            for k in self.criteria.keys():
                if not k.isdigit():
                    raise ValueError(f"Score question criteria keys must be integer strings (e.g. '0', '1', '2', '3', '4'), found '{k}'.")
        return self


class StructuredState(BaseModel):
    title: Optional[str] = None
    context: Optional[str] = None
    file_path: Optional[str] = None
    language: Optional[str] = None
    code_snippet: Optional[str] = None
    diff: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DecisionSample(BaseModel):
    id: str
    domain: DomainType
    tier: DataTier
    state: Union[StructuredState, str]
    questions: Dict[str, TypedQuestion]
    answers: Dict[str, Union[str, int, bool]]

    @model_validator(mode="after")
    def validate_answers_match_questions(self) -> "DecisionSample":
        q_keys = set(self.questions.keys())
        a_keys = set(self.answers.keys())

        if q_keys != a_keys:
            missing_in_answers = q_keys - a_keys
            extra_in_answers = a_keys - q_keys
            raise ValueError(
                f"Mismatch between questions and answers. "
                f"Missing in answers: {missing_in_answers}, Extra in answers: {extra_in_answers}"
            )

        for q_id, question in self.questions.items():
            ans = self.answers[q_id]
            if question.type == QuestionType.CHOICE:
                if not isinstance(ans, str):
                    raise ValueError(f"Answer for choice question '{q_id}' must be string, got {type(ans).__name__}")
                if question.criteria and ans not in question.criteria:
                    raise ValueError(f"Choice answer '{ans}' not found in criteria options: {list(question.criteria.keys())}")

            elif question.type == QuestionType.SCORE:
                if not isinstance(ans, int) or isinstance(ans, bool):
                    raise ValueError(f"Answer for score question '{q_id}' must be int, got {type(ans).__name__}")
                if str(ans) not in question.criteria:
                    raise ValueError(f"Score answer {ans} not in criteria keys: {list(question.criteria.keys())}")

            elif question.type == QuestionType.NOUL:
                if not isinstance(ans, bool):
                    raise ValueError(f"Answer for noul question '{q_id}' must be boolean (True/False), got {type(ans).__name__}")

        return self
