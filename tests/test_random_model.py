"""Tests cho RandomPolicy (Thành viên 3)."""
import unittest

from _helpers import RE, RECURSION_INTRODUCED, ctx, question_bank

from adaptive_learning.action_space import build_available_actions
from adaptive_learning.policies import BasePolicy, RandomPolicy
from adaptive_learning.schemas import LearnerState


class TestRandomPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = LearnerState(mastery={RE: 0.3}, attempted_question_ids=["REC_001"])
        cls.context = ctx(RE, RECURSION_INTRODUCED)
        cls.actions = build_available_actions(cls.state, question_bank(), cls.context)

    def _run(self, policy, n=30):
        return [policy.select_action(self.state, self.actions, self.context) for _ in range(n)]

    def test_same_seed_same_sequence(self):
        self.assertEqual(self._run(RandomPolicy(seed=42)), self._run(RandomPolicy(seed=42)))

    def test_reset_replays_sequence(self):
        policy = RandomPolicy(seed=3)
        first = self._run(policy)
        policy.reset()
        self.assertEqual(first, self._run(policy))

    def test_always_valid_and_covers_actions(self):
        decisions = self._run(RandomPolicy(seed=0), n=300)
        self.assertTrue(all(BasePolicy.validate(d, self.actions) for d in decisions))
        chosen = {(d.action_type, d.concept, d.difficulty) for d in decisions}
        self.assertEqual(len(chosen), len(self.actions))

    def test_works_without_context(self):
        decision = RandomPolicy(seed=1).select_action(self.state, self.actions)
        self.assertTrue(BasePolicy.validate(decision, self.actions))

    def test_empty_actions_raises(self):
        with self.assertRaises(ValueError):
            RandomPolicy(seed=1).select_action(self.state, ())


if __name__ == "__main__":
    unittest.main()
