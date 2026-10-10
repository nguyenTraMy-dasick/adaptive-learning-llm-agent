"""Seeded random baseline over legal actions."""
import random
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision
from .base import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_action(self, learner_state: LearnerState, available_actions: tuple[Action, ...]) -> PolicyDecision:
        if not available_actions:
            raise ValueError("available_actions must not be empty")
        return PolicyDecision.from_action(self.rng.choice(available_actions))
