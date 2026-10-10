"""Placeholder rule baseline; team-defined rules can be added without API changes."""
from adaptive_learning.schemas import Action, ActionType, LearnerState, PolicyDecision
from .base import BasePolicy

class RuleBasedPolicy(BasePolicy):
    def select_action(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        if not available_actions:
            raise ValueError("available_actions must not be empty")
        # Starter rule only: practice the lowest-mastery available concept.
        concepts = {a.concept for a in available_actions if a.concept is not None}
        weakest = min(concepts, key=lambda c: learner_state.mastery.get(c, 0.0), default=None)
        candidates = [a for a in available_actions if a.concept == weakest and a.action_type == ActionType.ASK]
        chosen = candidates[0] if candidates else available_actions[0]
        return PolicyDecision.from_action(chosen)
