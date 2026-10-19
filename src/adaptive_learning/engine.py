"""Thành viên 3: điểm tích hợp một lượt học; không đặt thuật toán mastery trong engine.

Luồng một lượt (run_turn):
1. Tạo available_actions từ learner state + DecisionContext (action_space.py).
2. Gọi policy.select_action(state, available_actions, context).
3. Python validate action (action_type, concept, difficulty); rationale không thay thế validation.
   Action không hợp lệ (hoặc policy raise lỗi) → KHÔNG thực thi, state giữ nguyên,
   TurnResult(valid=False, error=...) được ghi log.
4. Action hợp lệ → giao cho ActionExecutor (Thành viên 2): executor chọn câu hỏi từ
   QuestionBank cố định, lấy câu trả lời, chấm bằng answer key, cập nhật student model
   và trả TurnResult có updated_state.
5. Engine cập nhật DecisionContext (focus, concept đã giới thiệu, kết quả câu gần nhất)
   và ghi log.

Không để LLM tạo câu hỏi, chấm đáp án hoặc cập nhật mastery.

PHẦN CẦN THỐNG NHẤT VỚI THÀNH VIÊN 2: interface `ActionExecutor` bên dưới là *đề xuất*.
Engine chỉ cần executor có method `execute(decision, learner_state, respond)` trả TurnResult.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from adaptive_learning.action_space import build_available_actions
from adaptive_learning.policies.base import BasePolicy, DecisionContext
from adaptive_learning.question_bank import QuestionBank
from adaptive_learning.schemas import (
    Action,
    ActionType,
    LearnerState,
    PolicyDecision,
    Question,
    TurnResult,
)

# Hàm nhận câu hỏi và trả về lựa chọn của learner ("A".."D"); do người dùng/simulator cung cấp.
Responder = Callable[[Question], str]


@runtime_checkable
class ActionExecutor(Protocol):
    """ĐỀ XUẤT interface cho Thành viên 2 (cần hai bên xác nhận).

    execute() nhận một quyết định ĐÃ được engine validate và:
    - ASK: chọn câu chưa làm theo (concept, difficulty), gọi `respond(question)` để lấy
      câu trả lời, chấm bằng `question.correct_answer`, cập nhật student model.
    - HINT / EXPLAIN / REVIEW_PREREQUISITE / ADVANCE: trả feedback/nội dung, cập nhật
      lịch sử theo quy tắc của Thành viên 2.
    Trả về TurnResult với decision, valid=True, question/question_id/correct (nếu có),
    feedback và updated_state (LearnerState mới, không sửa state cũ).
    """

    def execute(
        self,
        decision: PolicyDecision,
        learner_state: LearnerState,
        respond: Responder | None = None,
    ) -> TurnResult: ...


@dataclass(frozen=True)
class TurnLog:
    """Một dòng log đủ để tái lập và phân tích một lượt.

    turn: số thứ tự lượt gọi policy (kể cả lượt invalid); context.step chỉ tăng khi
    action được thực thi.
    """

    turn: int
    policy: str
    context: DecisionContext
    state_before: LearnerState
    available_actions: tuple[Action, ...]
    result: TurnResult
    policy_trace: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "policy": self.policy,
            "context": self.context.to_dict(),
            "state_before": self.state_before.to_dict(),
            "available_actions": [a.to_dict() for a in self.available_actions],
            "result": self.result.to_dict(),
            "policy_trace": self.policy_trace,
        }


@dataclass(frozen=True)
class SessionResult:
    final_state: LearnerState
    final_context: DecisionContext
    logs: tuple[TurnLog, ...]
    stop_reason: str


class LearningEngine:
    def __init__(
        self,
        policy: BasePolicy,
        question_bank: QuestionBank,
        action_executor: ActionExecutor,
        mastery_goal: float = 0.85,
    ) -> None:
        if question_bank.concept_graph is None:
            raise ValueError("QuestionBank must have a concept_graph.")
        if not hasattr(action_executor, "execute"):
            raise TypeError("action_executor must provide execute(decision, learner_state, respond)")
        self.policy = policy
        self.question_bank = question_bank
        self.action_executor = action_executor
        self.mastery_goal = mastery_goal
        self.context = DecisionContext()
        self.logs: list[TurnLog] = []

    # --------------------------------------------------------------- helpers
    def reset(self, context: DecisionContext | None = None) -> None:
        self.context = context or DecisionContext()
        self.logs = []

    def available_actions(self, learner_state: LearnerState) -> tuple[Action, ...]:
        return build_available_actions(learner_state, self.question_bank, self.context)

    def _next_context(self, decision: PolicyDecision, result: TurnResult) -> DecisionContext:
        ctx = self.context
        focus = ctx.focus_concept
        if decision.action_type in (ActionType.ADVANCE, ActionType.REVIEW_PREREQUISITE):
            focus = decision.concept
        introduced = ctx.introduced_concepts + ((focus,) if focus else ())

        question_id, question_concept, correct = ctx.last_question_id, ctx.last_question_concept, ctx.last_correct
        if decision.action_type == ActionType.ASK and (result.question_id or result.question):
            question_id = result.question_id or result.question.question_id
            question = result.question or self.question_bank.get_question(question_id)
            question_concept = question.concept if question else decision.concept
            correct = result.correct

        return DecisionContext(
            focus_concept=focus,
            introduced_concepts=introduced,
            last_action=decision,
            last_question_id=question_id,
            last_question_concept=question_concept,
            last_correct=correct,
            step=ctx.step + 1,
        )

    def _record(self, state: LearnerState, actions: tuple[Action, ...], result: TurnResult) -> TurnResult:
        trace = getattr(self.policy, "last_trace", None)
        self.logs.append(
            TurnLog(
                turn=len(self.logs),
                policy=getattr(self.policy, "name", type(self.policy).__name__),
                context=self.context,
                state_before=state,
                available_actions=actions,
                result=result,
                policy_trace=dict(trace) if isinstance(trace, dict) and trace else None,
            )
        )
        return result

    # ------------------------------------------------------------------ turn
    def run_turn(self, learner_state: LearnerState, respond: Responder | None = None) -> TurnResult:
        """Chạy một lượt. Trả TurnResult; khi invalid thì updated_state là state cũ."""
        actions = self.available_actions(learner_state)
        if not actions:
            raise RuntimeError("No available actions: every question has been used.")

        try:
            decision = self.policy.select_action(learner_state, actions, self.context)
        except Exception as exc:  # LLM lỗi định dạng, lỗi mạng... → bước invalid
            invalid = TurnResult(
                decision=None,  # policy không trả được quyết định nào
                valid=False,
                updated_state=learner_state,
                error=f"policy error: {type(exc).__name__}: {exc}",
            )
            return self._record(learner_state, actions, invalid)

        if not BasePolicy.validate(decision, actions):
            invalid = TurnResult(
                decision=decision,
                valid=False,
                updated_state=learner_state,
                error="decision is not in available_actions",
            )
            return self._record(learner_state, actions, invalid)

        result = self.action_executor.execute(decision, learner_state, respond)
        if result.updated_state is None:
            result = replace(result, updated_state=learner_state)
        self._record(learner_state, actions, result)
        self.context = self._next_context(decision, result)
        return result

    # --------------------------------------------------------------- session
    def goal_reached(self, learner_state: LearnerState) -> bool:
        concepts = self.question_bank.get_all_concepts()
        return all(learner_state.mastery.get(c, 0.0) >= self.mastery_goal for c in concepts)

    def run_session(
        self,
        initial_state: LearnerState,
        respond: Responder | None = None,
        max_steps: int = 100,
        max_consecutive_invalid: int = 3,
    ) -> SessionResult:
        """Chạy nhiều lượt tới khi đạt mục tiêu, hết action, hết bước hoặc lỗi liên tiếp."""
        state = initial_state
        invalid_streak = 0
        stop_reason = "max_steps"
        for _ in range(max_steps):
            if self.goal_reached(state):
                stop_reason = "goal_reached"
                break
            if not self.available_actions(state):
                stop_reason = "no_actions"
                break
            result = self.run_turn(state, respond)
            if result.valid:
                invalid_streak = 0
                state = result.updated_state or state
            else:
                invalid_streak += 1
                if invalid_streak >= max_consecutive_invalid:
                    stop_reason = "too_many_invalid"
                    break
        else:
            if self.goal_reached(state):
                stop_reason = "goal_reached"
        return SessionResult(state, self.context, tuple(self.logs), stop_reason)

    def write_log(self, path: str | Path) -> Path:
        """Ghi log dạng JSON Lines (mỗi lượt một dòng)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for entry in self.logs:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return out


Engine = LearningEngine
