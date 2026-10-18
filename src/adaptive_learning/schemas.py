"""HỢP ĐỒNG CHUNG — xem TEAM_TASKS.md trước khi thay đổi schema.

Cả ba policy và engine phải dùng chung các kiểu dưới đây. Giữ schema nhỏ,
serializable, không đưa logic chọn policy/chấm điểm vào file này.
Đã chốt:
- Validate mastery trong đoạn [0.0, 1.0] (số thực hữu hạn; không nhận bool, NaN, inf).
- `LearnerState.mastery` là mapping chỉ đọc (FrozenMastery); muốn cập nhật thì tạo
  LearnerState mới, ví dụ `LearnerState(mastery={**state.mastery, concept: 0.6}, ...)`.
- Biểu diễn danh sách trong state dưới dạng immutable tuple khi chạy, serialize thành list khi ra JSON.
- Difficulty thống nhất với CSV: Easy, Medium, Hard (Difficulty enum và ALLOWED_DIFFICULTIES).
- Prerequisite concept giữ nguyên tên cột CSV `prerequisite_concept` (số ít),
  đồng thời cung cấp property `prerequisites` (tuple) để tiện truy xuất danh sách.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Difficulty(str, Enum):
    """Mức độ khó của câu hỏi, chuẩn hóa theo CSV."""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


ALLOWED_DIFFICULTIES: frozenset[str] = frozenset({d.value for d in Difficulty})


class ActionType(str, Enum):
    """Các action chung mà policy được phép đề xuất."""
    ASK = "ASK"
    HINT = "HINT"
    EXPLAIN = "EXPLAIN"
    REVIEW_PREREQUISITE = "REVIEW_PREREQUISITE"
    ADVANCE = "ADVANCE"


class FrozenMastery(dict):
    """dict chỉ đọc dùng cho LearnerState.mastery.

    Vẫn là dict (so sánh, json.dumps, dict(...) đều dùng được) nhưng mọi thao tác
    sửa đều báo TypeError, để policy không thể vô tình sửa state dùng chung.
    """

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "LearnerState.mastery is read-only; create a new LearnerState to update mastery."
        )

    __setitem__ = _readonly
    __delitem__ = _readonly
    __ior__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[Any, ...]:
        # Cho phép copy/deepcopy/pickle mà không đi qua __setitem__.
        return (FrozenMastery, (dict(self),))


def _validate_mastery(mastery: Mapping[str, float]) -> dict[str, float]:
    """Kiểm tra và chuẩn hóa giá trị mastery trong khoảng [0.0, 1.0]."""
    cleaned: dict[str, float] = {}
    for concept, score in mastery.items():
        if not isinstance(concept, str):
            raise TypeError(f"Concept key must be str, got {type(concept).__name__}: {concept!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(
                f"Mastery score for '{concept}' must be float or int, got {type(score).__name__}: {score!r}"
            )
        score_f = float(score)
        if not math.isfinite(score_f):
            raise ValueError(f"Mastery score for '{concept}' must be a finite number, got {score_f}")
        if score_f < 0.0 or score_f > 1.0:
            raise ValueError(f"Mastery score for '{concept}' must be in [0.0, 1.0], got {score_f}")
        cleaned[concept] = score_f
    return cleaned


def _validate_difficulty(difficulty: str | Difficulty | None) -> str | None:
    """Kiểm tra độ khó thuộc ALLOWED_DIFFICULTIES hoặc None."""
    if difficulty is None:
        return None
    val = difficulty.value if isinstance(difficulty, Difficulty) else str(difficulty).strip()
    if val not in ALLOWED_DIFFICULTIES:
        raise ValueError(
            f"Invalid difficulty: {difficulty!r}. Allowed values: {sorted(ALLOWED_DIFFICULTIES)}"
        )
    return val


def _coerce_action_type(action_type: ActionType | str) -> ActionType:
    """Chuyển đổi chuỗi thành ActionType enum hợp lệ."""
    if isinstance(action_type, ActionType):
        return action_type
    if isinstance(action_type, str):
        val = action_type.strip().upper()
        if val in ActionType.__members__:
            return ActionType[val]
        try:
            return ActionType(val)
        except ValueError:
            pass
    raise ValueError(
        f"Invalid action_type: {action_type!r}. Allowed: {[a.value for a in ActionType]}"
    )


@dataclass(frozen=True)
class LearnerState:
    """Snapshot learner state; immutable, policy không được mutate object này."""
    mastery: Mapping[str, float] = field(default_factory=dict)
    recent_errors: tuple[str, ...] = ()
    recent_actions: tuple[str, ...] = ()
    attempted_question_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cleaned_mastery = FrozenMastery(_validate_mastery(self.mastery))
        object.__setattr__(self, "mastery", cleaned_mastery)
        object.__setattr__(self, "recent_errors", tuple(str(x) for x in self.recent_errors))
        object.__setattr__(self, "recent_actions", tuple(str(x) for x in self.recent_actions))
        object.__setattr__(self, "attempted_question_ids", tuple(str(x) for x in self.attempted_question_ids))

    def to_dict(self) -> dict[str, Any]:
        """Serialize sang JSON-compatible dictionary."""
        return {
            "mastery": dict(self.mastery),
            "recent_errors": list(self.recent_errors),
            "recent_actions": list(self.recent_actions),
            "attempted_question_ids": list(self.attempted_question_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LearnerState:
        """Khởi tạo LearnerState từ dictionary (JSON parse)."""
        return cls(
            mastery=data.get("mastery", {}),
            recent_errors=tuple(data.get("recent_errors", ())),
            recent_actions=tuple(data.get("recent_actions", ())),
            attempted_question_ids=tuple(data.get("attempted_question_ids", ())),
        )


@dataclass(frozen=True)
class Action:
    """Action hợp lệ được engine tạo trước khi gọi policy."""
    action_type: ActionType
    concept: str | None = None
    difficulty: str | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        act = _coerce_action_type(self.action_type)
        diff = _validate_difficulty(self.difficulty)
        conc = str(self.concept).strip() if self.concept is not None else None
        object.__setattr__(self, "action_type", act)
        object.__setattr__(self, "concept", conc)
        object.__setattr__(self, "difficulty", diff)
        object.__setattr__(self, "rationale", str(self.rationale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "concept": self.concept,
            "difficulty": self.difficulty,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Action:
        return cls(
            action_type=data["action_type"],
            concept=data.get("concept"),
            difficulty=data.get("difficulty"),
            rationale=data.get("rationale", ""),
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Output thống nhất của cả ba policy; rationale chỉ phục vụ phân tích."""
    action_type: ActionType
    concept: str | None = None
    difficulty: str | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        act = _coerce_action_type(self.action_type)
        diff = _validate_difficulty(self.difficulty)
        conc = str(self.concept).strip() if self.concept is not None else None
        object.__setattr__(self, "action_type", act)
        object.__setattr__(self, "concept", conc)
        object.__setattr__(self, "difficulty", diff)
        object.__setattr__(self, "rationale", str(self.rationale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "concept": self.concept,
            "difficulty": self.difficulty,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyDecision:
        return cls(
            action_type=data["action_type"],
            concept=data.get("concept"),
            difficulty=data.get("difficulty"),
            rationale=data.get("rationale", ""),
        )


@dataclass(frozen=True)
class Question:
    """Một câu hỏi trắc nghiệm trong QuestionBank, khớp chính xác các cột CSV."""
    question_id: str
    concept: str
    prerequisite_concept: str
    question: str
    option_A: str
    option_B: str
    option_C: str
    option_D: str
    difficulty: str
    correct_answer: str
    common_error_type: str = ""
    option_error_mapping: str = ""
    explanation: str = ""

    def __post_init__(self) -> None:
        diff = _validate_difficulty(self.difficulty)
        if diff is None:
            raise ValueError(
                f"Question difficulty cannot be None, must be one of {sorted(ALLOWED_DIFFICULTIES)}"
            )
        ans = str(self.correct_answer).strip().upper()
        if ans not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid correct_answer: {self.correct_answer!r}. Must be A, B, C, or D.")
        object.__setattr__(self, "question_id", str(self.question_id).strip())
        object.__setattr__(self, "concept", str(self.concept).strip())
        object.__setattr__(self, "prerequisite_concept", str(self.prerequisite_concept).strip())
        object.__setattr__(self, "question", str(self.question).strip())
        object.__setattr__(self, "option_A", str(self.option_A).strip())
        object.__setattr__(self, "option_B", str(self.option_B).strip())
        object.__setattr__(self, "option_C", str(self.option_C).strip())
        object.__setattr__(self, "option_D", str(self.option_D).strip())
        object.__setattr__(self, "difficulty", diff)
        object.__setattr__(self, "correct_answer", ans)
        object.__setattr__(self, "common_error_type", str(self.common_error_type).strip())
        object.__setattr__(self, "option_error_mapping", str(self.option_error_mapping).strip())
        object.__setattr__(self, "explanation", str(self.explanation).strip())

    @property
    def options(self) -> dict[str, str]:
        """Map các lựa chọn đáp án dạng dictionary {'A': ..., 'B': ...}."""
        return {
            "A": self.option_A,
            "B": self.option_B,
            "C": self.option_C,
            "D": self.option_D,
        }

    @property
    def prerequisites(self) -> tuple[str, ...]:
        """Tách prerequisite_concept thành tuple các tên concept (bỏ qua 'None')."""
        raw = self.prerequisite_concept
        if not raw or raw.lower() in {"none", "null", ""}:
            return ()
        parts = [p.strip() for p in raw.split(";") if p.strip() and p.strip().lower() != "none"]
        return tuple(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "concept": self.concept,
            "prerequisite_concept": self.prerequisite_concept,
            "question": self.question,
            "option_A": self.option_A,
            "option_B": self.option_B,
            "option_C": self.option_C,
            "option_D": self.option_D,
            "difficulty": self.difficulty,
            "correct_answer": self.correct_answer,
            "common_error_type": self.common_error_type,
            "option_error_mapping": self.option_error_mapping,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Question:
        return cls(
            question_id=data["question_id"],
            concept=data["concept"],
            prerequisite_concept=data.get("prerequisite_concept", "None"),
            question=data["question"],
            option_A=data["option_A"],
            option_B=data["option_B"],
            option_C=data["option_C"],
            option_D=data["option_D"],
            difficulty=data["difficulty"],
            correct_answer=data["correct_answer"],
            common_error_type=data.get("common_error_type", ""),
            option_error_mapping=data.get("option_error_mapping", ""),
            explanation=data.get("explanation", ""),
        )


@dataclass(frozen=True)
class TurnResult:
    """Kết quả khái quát một lượt; chốt fields cuối cùng khi tích hợp."""
    decision: PolicyDecision
    valid: bool
    question_id: str | None = None
    question: Question | None = None
    feedback: str | None = None
    correct: bool | None = None
    updated_state: LearnerState | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict() if isinstance(self.decision, PolicyDecision) else self.decision,
            "valid": self.valid,
            "question_id": self.question_id,
            "question": self.question.to_dict() if isinstance(self.question, Question) else self.question,
            "feedback": self.feedback,
            "correct": self.correct,
            "updated_state": self.updated_state.to_dict() if isinstance(self.updated_state, LearnerState) else None,
            "error": self.error,
        }
