"""Thành viên 3: implement baseline chọn ngẫu nhiên có seed để tái lập.

Input/output phải đúng BasePolicy; chỉ chọn từ available_actions. Không sinh
question mới, không chấm response và không cập nhật mastery.
TODO: khởi tạo random.Random(seed), xử lý action list rỗng, trả PolicyDecision.
"""
from adaptive_learning.policies.base import BasePolicy
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision


class RandomPolicy(BasePolicy):
    def __init__(self, seed: int | None = None):
        # TODO: lưu random generator có seed.
        raise NotImplementedError

    def select_action(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        # TODO: chọn đều trong các action hợp lệ.
        raise NotImplementedError
