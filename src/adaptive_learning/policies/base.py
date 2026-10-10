"""One interface shared by all pedagogical policies."""
from abc import ABC, abstractmethod
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision


class BasePolicy(ABC):
    @abstractmethod
    def select_action(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        """Return one decision from the supplied legal action set."""

    @staticmethod
    def validate(decision: PolicyDecision, available_actions: tuple[Action, ...]) -> bool:
        return any((decision.action_type, decision.concept, decision.difficulty) ==
                   (action.action_type, action.concept, action.difficulty)
                   for action in available_actions)
