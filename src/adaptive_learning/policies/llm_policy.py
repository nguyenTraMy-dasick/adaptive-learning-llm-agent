"""Thành viên 3: adapter cho LLM; model chỉ đề xuất action.

TODO:
- Inject model client/callable để dễ thay provider và mock khi đánh giá.
- Gửi learner_state cùng danh sách available_actions đã serialize.
- Yêu cầu output có cấu trúc theo PolicyDecision.
- Parse output; engine vẫn là nơi kiểm tra cuối cùng action có hợp lệ không.
- Không gửi đáp án đúng nếu không cần cho quyết định; tuyệt đối không yêu cầu
  model chấm câu trả lời hoặc cập nhật learner mastery.
"""
from adaptive_learning.policies.base import BasePolicy
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision


class LLMPolicy(BasePolicy):
    def __init__(self, model_client):
        # TODO: lưu client/cấu hình cần thiết; không hard-code nhà cung cấp.
        raise NotImplementedError

    def select_action(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        # TODO: serialize -> gọi model -> parse thành PolicyDecision.
        raise NotImplementedError
