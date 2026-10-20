"""Execute validated actions using the fixed QuestionBank and StudentModel."""
from __future__ import annotations

import random
from typing import Any, Callable

from adaptive_learning.question_bank import QuestionBank
from adaptive_learning.schemas import ActionType, LearnerState, PolicyDecision, Question, TurnResult
from adaptive_learning.student_model import StudentModel

Responder = Callable[[Question], Any]


class ActionExecutor:
    """Reference executor for ASK, support, prerequisite review and advance."""

    def __init__(
        self,
        question_bank: QuestionBank,
        student_model: StudentModel | None = None,
        *,
        question_strategy: str = "first",
        rng: random.Random | None = None,
    ) -> None:
        if question_strategy not in {"first", "random"}:
            raise ValueError("question_strategy must be 'first' or 'random'.")
        self.question_bank = question_bank
        self.student_model = student_model or StudentModel()
        self.question_strategy = question_strategy
        self.rng = rng

    @staticmethod
    def _validate_decision(decision: PolicyDecision) -> None:
        if not isinstance(decision, PolicyDecision):
            raise TypeError("decision must be a PolicyDecision.")
        if not decision.concept:
            raise ValueError(f"{decision.action_type.value} requires a concept.")
        if decision.action_type == ActionType.ASK and not decision.difficulty:
            raise ValueError("ASK requires a difficulty.")

    def _support_question(self, concept: str, state: LearnerState) -> Question | None:
        for qid in reversed(state.attempted_question_ids):
            question = self.question_bank.get_question(qid)
            if question is not None and question.concept == concept:
                return question
        return None

    def _feedback(self, decision: PolicyDecision, state: LearnerState) -> str:
        concept = decision.concept or ""
        graph = self.question_bank.concept_graph
        info = graph.get_concept(concept) if graph else None
        if decision.action_type in {ActionType.EXPLAIN, ActionType.REVIEW_PREREQUISITE}:
            return (info or {}).get("description", f"Review the key ideas of {concept}.")
        if decision.action_type == ActionType.HINT:
            question = self._support_question(concept, state)
            if question:
                return f"Focus on the distinction tested here: {question.common_error_type or 'check each option carefully'}."
            return f"Review the key idea behind {concept} before answering."
        return f"Now focusing on {concept}."

    def execute(
        self,
        decision: PolicyDecision,
        learner_state: LearnerState,
        respond: Responder | None = None,
    ) -> TurnResult:
        self._validate_decision(decision)
        if decision.action_type == ActionType.ASK:
            question = self.question_bank.get_next_question(
                decision.concept,
                decision.difficulty,
                learner_state.attempted_question_ids,
                strategy=self.question_strategy,
                rng=self.rng,
            )
            if question is None:
                return TurnResult(
                    decision=decision,
                    valid=False,
                    updated_state=learner_state,
                    error=f"No unanswered question for {decision.concept!r} at {decision.difficulty!r}.",
                )
            if respond is None:
                return TurnResult(
                    decision=decision,
                    valid=False,
                    updated_state=learner_state,
                    error="ASK requires a responder; the executor cannot infer or reveal the answer.",
                )
            answer = respond(question)
            updated_state, correct = self.student_model.update_state(learner_state, question, answer)
            feedback = f"{'Correct.' if correct else 'Incorrect.'} {question.explanation}".strip()
            return TurnResult(
                decision=decision,
                valid=True,
                question_id=question.question_id,
                question=question,
                feedback=feedback,
                correct=correct,
                updated_state=updated_state,
            )

        updated_state = self.student_model.record_action(
            learner_state, f"{decision.action_type.value}:{decision.concept}"
        )
        return TurnResult(
            decision=decision,
            valid=True,
            feedback=self._feedback(decision, learner_state),
            updated_state=updated_state,
        )


DefaultActionExecutor = ActionExecutor
