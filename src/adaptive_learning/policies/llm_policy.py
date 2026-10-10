"""LLM adapter skeleton; model invocation is injected to avoid provider lock-in."""
import json
from typing import Callable, Any
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision
from .base import BasePolicy

class LLMPolicy(BasePolicy):
    def __init__(self, selector: Callable[[dict[str, Any]], Any]):
        self.selector = selector

    def select_action(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        if not available_actions:
            raise ValueError("available_actions must not be empty")
        payload = {"learner_state": {"mastery": dict(learner_state.mastery),
                   "recent_errors": list(learner_state.recent_errors),
                   "recent_actions": list(learner_state.recent_actions),
                   "attempted_question_ids": list(learner_state.attempted_question_ids)},
                   "available_actions": [a.__dict__ for a in available_actions]}
        proposed = self.selector(payload)
        if isinstance(proposed, str):
            proposed = json.loads(proposed)
        if isinstance(proposed, int):
            if not 0 <= proposed < len(available_actions):
                raise ValueError("LLM selected an out-of-range action")
            return PolicyDecision.from_action(available_actions[proposed])
        decision = PolicyDecision(**proposed)
        if not BasePolicy.validate(decision, available_actions):
            raise ValueError("LLM selected an action outside available_actions")
        return decision
