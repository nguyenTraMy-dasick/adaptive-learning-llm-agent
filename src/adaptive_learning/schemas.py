"""Shared data contracts for policy and engine integration."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Any


class ActionType(str, Enum):
    ASK = "ASK"
    HINT = "HINT"
    EXPLAIN = "EXPLAIN"
    REVIEW_PREREQUISITE = "REVIEW_PREREQUISITE"
    ADVANCE = "ADVANCE"


@dataclass(frozen=True)
class LearnerState:
    mastery: Mapping[str, float] = field(default_factory=dict)
    recent_errors: tuple[str, ...] = ()
    recent_actions: tuple[str, ...] = ()
    attempted_question_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    concept: str | None = None
    difficulty: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class PolicyDecision:
    action_type: ActionType
    concept: str | None = None
    difficulty: str | None = None
    rationale: str = ""

    @classmethod
    def from_action(cls, action: Action) -> "PolicyDecision":
        return cls(action.action_type, action.concept, action.difficulty, action.rationale)

    def as_dict(self) -> dict[str, Any]:
        return {"action_type": self.action_type.value, "concept": self.concept,
                "difficulty": self.difficulty, "rationale": self.rationale}


@dataclass(frozen=True)
class TurnResult:
    decision: PolicyDecision
    valid: bool
    question_id: str | None = None
    question: Any = None
    feedback: Any = None
    correct: bool | None = None
    updated_state: LearnerState | None = None
    error: str | None = None
