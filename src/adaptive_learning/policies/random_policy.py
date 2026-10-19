"""Thành viên 3: baseline chọn ngẫu nhiên có seed để tái lập.

Baseline A (không cá nhân hóa): chọn đều một action trong available_actions,
không nhìn learner state hay context. Không sinh câu hỏi, không chấm đáp án,
không cập nhật mastery.

Lưu ý: chọn đều theo *action* (không phải theo loại action), nên loại có nhiều
biến thể (ASK có tới 3 độ khó, ADVANCE có nhiều concept) được chọn thường hơn.
"""
from __future__ import annotations

import random

from adaptive_learning.policies.base import BasePolicy, DecisionContext, to_decision
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Đặt lại generator (dùng seed cũ nếu không truyền) để chạy lại y hệt."""
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)

    def select_action(
        self,
        learner_state: LearnerState,
        available_actions: tuple[Action, ...],
        context: DecisionContext | None = None,
    ) -> PolicyDecision:
        self._require_actions(available_actions)
        action = self._rng.choice(list(available_actions))
        return to_decision(action, rationale="random choice")
