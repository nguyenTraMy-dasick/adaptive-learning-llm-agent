"""High-level learning turn orchestration; domain logic is injected later."""
from typing import Any, Callable
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision, TurnResult
from adaptive_learning.policies.base import BasePolicy


class LearningEngine:
    """Validates policy proposals and delegates deterministic work to Python components.

    Inject ``question_selector``, ``grader`` and ``state_updater`` during integration.
    Keeping these seams explicit lets the team settle contracts before implementing
    a particular mastery model or instructional content strategy.
    """
    def __init__(self, policy: BasePolicy, question_selector: Callable[..., Any] | None = None,
                 grader: Callable[..., bool] | None = None,
                 state_updater: Callable[..., LearnerState] | None = None) -> None:
        self.policy = policy
        self.question_selector = question_selector
        self.grader = grader
        self.state_updater = state_updater

    def choose(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        if not available_actions:
            raise ValueError("available_actions must not be empty")
        decision = self.policy.select_action(learner_state, available_actions)
        if not BasePolicy.validate(decision, available_actions):
            raise ValueError("Policy proposed an action outside available_actions")
        return decision

    def run_turn(self, learner_state: LearnerState, available_actions: tuple[Action, ...],
                 response: str | None = None) -> TurnResult:
        decision = self.choose(learner_state, available_actions)
        if decision.action_type.value == "ADVANCE":
            return TurnResult(decision, True, updated_state=learner_state)
        if decision.action_type.value != "ASK":
            # HINT/EXPLAIN/REVIEW content providers are an integration seam.
            return TurnResult(decision, True, updated_state=learner_state)
        if self.question_selector is None:
            return TurnResult(decision, True, updated_state=learner_state)
        question = self.question_selector(decision, learner_state)
        qid = question.get("question_id") if isinstance(question, dict) else getattr(question, "question_id", None)
        if response is None:
            return TurnResult(decision, True, question_id=qid, question=question, updated_state=learner_state)
        if self.grader is None or self.state_updater is None:
            raise RuntimeError("Grader and state_updater are required to process a response")
        # Correctness is computed by Python; only this result reaches the student-model updater.
        correct = bool(self.grader(question, response))
        updated = self.state_updater(learner_state, question, correct)
        return TurnResult(decision, True, question_id=qid, question=question, correct=correct, updated_state=updated)


Engine = LearningEngine
