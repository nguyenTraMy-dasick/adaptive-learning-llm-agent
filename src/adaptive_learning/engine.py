"""Điểm tích hợp một lượt học; không đặt thuật toán mastery trong engine.

Thành viên 3 implement sau khi thống nhất API với thành viên 1 (QuestionBank)
và thành viên 2 (ActionExecutor/StudentModel).
Luồng cần giữ:
1. Nhận learner state và tạo available_actions hợp lệ.
2. Gọi policy.select_action(state, available_actions).
3. Python validate action, concept và difficulty; rationale không thay thế validation.
4. Python thực thi action; với ASK, chọn câu hỏi từ QuestionBank cố định.
5. Nhận response; Python grader đối chiếu answer key.
6. StudentModel cập nhật mastery và lịch sử; engine ghi log TurnResult.

Không để LLM tạo câu hỏi, chấm đáp án hoặc cập nhật mastery.
TODO: thống nhất hành vi action không phải ASK, hết câu hỏi, và ADVANCE.
"""
from adaptive_learning.policies.base import BasePolicy
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision, TurnResult


class LearningEngine:
    def __init__(self, policy: BasePolicy, question_bank, action_executor, student_model):
        # TODO: lưu dependencies; chốt interface với các thành viên phụ trách.
        raise NotImplementedError

    def run_turn(self, learner_state: LearnerState, available_actions: tuple[Action, ...], response=None) -> TurnResult:
        # TODO: implement orchestration theo luồng ở module docstring.
        raise NotImplementedError


Engine = LearningEngine
