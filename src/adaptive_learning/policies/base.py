"""Interface duy nhất cho Random, Rule-based và LLM policy.

Thành viên 3 phụ trách. Các policy nhận cùng learner state, cùng available_actions
và cùng DecisionContext; chỉ khác thuật toán lựa chọn. Policy không gọi executor,
không chấm đáp án và không cập nhật mastery.

DecisionContext là phần "trạng thái phiên học" mà engine (Thành viên 3) tự theo dõi:
concept đang học (focus), các concept đã được giới thiệu, action trước đó và kết quả
câu hỏi gần nhất. Nhờ vậy policy không phải đoán từ chuỗi tự do trong
`recent_actions` / `recent_errors` (định dạng hai trường này do Thành viên 2 chốt).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from adaptive_learning.schemas import Action, ActionType, LearnerState, PolicyDecision


@dataclass(frozen=True)
class DecisionContext:
    """Bối cảnh phiên học mà engine cung cấp cho mọi policy ở mỗi bước.

    Attributes:
        focus_concept: concept đang học; None khi phiên vừa bắt đầu.
        introduced_concepts: các concept đã từng được chọn làm focus (đã "mở").
        last_action: quyết định đã thực thi ở bước trước (None nếu chưa có).
        last_question_id: câu hỏi được hỏi gần nhất (None nếu chưa hỏi).
        last_question_concept: concept của câu hỏi đó.
        last_correct: kết quả chấm câu hỏi gần nhất (None nếu chưa có).
        step: số bước đã thực thi trong phiên.
    """

    focus_concept: str | None = None
    introduced_concepts: tuple[str, ...] = ()
    last_action: PolicyDecision | None = None
    last_question_id: str | None = None
    last_question_concept: str | None = None
    last_correct: bool | None = None
    step: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "introduced_concepts", tuple(dict.fromkeys(self.introduced_concepts)))
        if self.last_action is not None and not isinstance(self.last_action, PolicyDecision):
            raise TypeError("last_action must be a PolicyDecision or None")
        if self.step < 0:
            raise ValueError("step must be >= 0")

    @property
    def just_answered_wrong(self) -> bool:
        """True nếu action NGAY TRƯỚC là ASK trên focus concept và bị trả lời sai.

        Khác `last_answer_wrong_on_focus`: sau một action khác (EXPLAIN, REVIEW, ADVANCE...)
        thì không còn là "vừa sai", tránh lặp lại cùng một phản ứng mãi.
        """
        last = self.last_action
        return (
            self.last_answer_wrong_on_focus
            and last is not None
            and last.action_type.value == "ASK"
            and last.concept == self.focus_concept
        )

    @property
    def last_answer_wrong_on_focus(self) -> bool:
        """True nếu câu hỏi gần nhất thuộc focus concept và bị trả lời sai."""
        return (
            self.last_correct is False
            and self.focus_concept is not None
            and self.last_question_concept == self.focus_concept
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_concept": self.focus_concept,
            "introduced_concepts": list(self.introduced_concepts),
            "last_action": self.last_action.to_dict() if self.last_action else None,
            "last_question_id": self.last_question_id,
            "last_question_concept": self.last_question_concept,
            "last_correct": self.last_correct,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> DecisionContext:
        data = data or {}
        last_action = data.get("last_action")
        return cls(
            focus_concept=data.get("focus_concept"),
            introduced_concepts=tuple(data.get("introduced_concepts", ())),
            last_action=PolicyDecision.from_dict(last_action) if last_action else None,
            last_question_id=data.get("last_question_id"),
            last_question_concept=data.get("last_question_concept"),
            last_correct=data.get("last_correct"),
            step=int(data.get("step", 0)),
        )


def action_key(action: Action | PolicyDecision) -> tuple[ActionType, str | None, str | None]:
    """Khóa so khớp action: (action_type, concept, difficulty); bỏ qua rationale."""
    return (action.action_type, action.concept, action.difficulty)


def to_decision(action: Action, rationale: str = "") -> PolicyDecision:
    """Chuyển một Action trong available_actions thành PolicyDecision."""
    return PolicyDecision(
        action_type=action.action_type,
        concept=action.concept,
        difficulty=action.difficulty,
        rationale=rationale,
    )


class BasePolicy(ABC):
    """Lớp cha của mọi policy.

    Hợp đồng:
    - Chỉ trả về một action có trong `available_actions` (so khớp action_type,
      concept, difficulty; rationale không tính).
    - Không sửa learner_state, available_actions hay context.
    - Raise ValueError nếu available_actions rỗng.
    """

    name: str = "base"

    @abstractmethod
    def select_action(
        self,
        learner_state: LearnerState,
        available_actions: tuple[Action, ...],
        context: DecisionContext | None = None,
    ) -> PolicyDecision:
        """Chọn một action trong available_actions."""
        raise NotImplementedError

    @staticmethod
    def validate(decision: PolicyDecision, available_actions: Iterable[Action]) -> bool:
        """True nếu decision trùng một action hợp lệ (bỏ qua rationale)."""
        if not isinstance(decision, PolicyDecision):
            return False
        key = action_key(decision)
        return any(action_key(a) == key for a in available_actions)

    @staticmethod
    def _require_actions(available_actions: tuple[Action, ...]) -> None:
        if not available_actions:
            raise ValueError("available_actions is empty; the engine must not call a policy without actions.")
