"""Interface duy nhất cho Random, Rule-based và LLM policy.

Thành viên 3 implement policy cụ thể. Các policy nhận cùng learner state và
available_actions; chỉ khác thuật toán lựa chọn. Không gọi executor/chấm đáp án.
"""
from abc import ABC, abstractmethod
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision


class BasePolicy(ABC):
    @abstractmethod
    def select_action(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        """Chọn một action trong available_actions; không sửa state/actions."""
        raise NotImplementedError

    @staticmethod
    def validate(decision: PolicyDecision, available_actions: tuple[Action, ...]) -> bool:
        """TODO: so khớp action_type, concept, difficulty; bỏ qua rationale."""
        raise NotImplementedError
