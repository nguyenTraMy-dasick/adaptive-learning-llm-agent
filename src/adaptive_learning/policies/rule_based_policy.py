"""Thành viên 3: implement baseline luật do nhóm thống nhất.

Ghi luật rõ ràng trong docstring/tài liệu để decision có thể giải thích và so
sánh. Chỉ chọn từ available_actions; không tự thêm action vào action space.
TODO: chốt thresholds/ưu tiên remediaton, mastery, độ khó, prerequisite.
"""
from adaptive_learning.policies.base import BasePolicy
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision


class RuleBasedPolicy(BasePolicy):
    def select_action(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        # TODO: áp dụng luật đã được nhóm duyệt rồi trả PolicyDecision.
        raise NotImplementedError
