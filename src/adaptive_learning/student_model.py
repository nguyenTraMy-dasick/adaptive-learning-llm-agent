"""Deterministic learner model used by every policy during a learning session."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from adaptive_learning.schemas import LearnerState, Question


@dataclass(frozen=True)
class StudentModel:
    """Update mastery with a fixed exponential moving average."""

    learning_rate: float = 0.30
    initial_mastery: float = 0.0
    history_limit: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise ValueError("learning_rate must be a finite number.")
        if not isfinite(float(self.learning_rate)) or not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if isinstance(self.initial_mastery, bool) or not isinstance(self.initial_mastery, (int, float)):
            raise ValueError("initial_mastery must be a finite number.")
        if not isfinite(float(self.initial_mastery)) or not 0.0 <= self.initial_mastery <= 1.0:
            raise ValueError("initial_mastery must be in [0, 1].")
        if isinstance(self.history_limit, bool) or not isinstance(self.history_limit, int) or self.history_limit <= 0:
            raise ValueError("history_limit must be a positive integer.")

    @staticmethod
    def grade_answer(question: Question, answer: Any) -> bool:
        """Grade only against the answer key from QuestionBank."""
        return answer is not None and str(answer).strip().upper() == question.correct_answer

    def update_mastery(self, mastery: float, correct: bool) -> float:
        """Return ``old + learning_rate * (outcome - old)``, bounded to [0, 1]."""
        target = 1.0 if correct else 0.0
        updated = float(mastery) + self.learning_rate * (target - float(mastery))
        return min(1.0, max(0.0, updated))

    def update_state(self, state: LearnerState, question: Question, answer: Any) -> tuple[LearnerState, bool]:
        """Grade one question and return a new state plus its correctness."""
        correct = self.grade_answer(question, answer)
        old_mastery = state.mastery.get(question.concept, self.initial_mastery)
        mastery = self.update_mastery(old_mastery, correct)
        action = f"ASK:{question.question_id}:{question.concept}:{'correct' if correct else 'wrong'}"
        actions = (state.recent_actions + (action,))[-self.history_limit :]
        errors = state.recent_errors
        if not correct:
            errors = (errors + (question.common_error_type or "incorrect_answer",))[-self.history_limit :]
        attempted = state.attempted_question_ids
        if question.question_id not in attempted:
            attempted = attempted + (question.question_id,)
        return LearnerState(
            mastery={**state.mastery, question.concept: mastery},
            recent_errors=errors,
            recent_actions=actions,
            attempted_question_ids=attempted,
        ), correct

    def record_action(self, state: LearnerState, action: str) -> LearnerState:
        """Record a non-question action without changing mastery."""
        return LearnerState(
            mastery=state.mastery,
            recent_errors=state.recent_errors,
            recent_actions=(state.recent_actions + (action,))[-self.history_limit :],
            attempted_question_ids=state.attempted_question_ids,
        )