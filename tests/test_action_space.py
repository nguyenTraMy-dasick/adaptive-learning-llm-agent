"""Tests cho action_space.build_available_actions (Thành viên 3)."""
import unittest

from _helpers import BO, CO, DT, EX, FU, RE, RECURSION_INTRODUCED, SC, V, ctx, question_bank

from adaptive_learning.action_space import build_available_actions, introduced_from_attempts
from adaptive_learning.policies.base import DecisionContext
from adaptive_learning.schemas import ActionType, LearnerState


def keys(actions):
    return [(a.action_type.value, a.concept, a.difficulty) for a in actions]


class TestActionSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qb = question_bank()

    def test_session_start_only_root_concepts(self):
        actions = build_available_actions(LearnerState(), self.qb, DecisionContext())
        self.assertEqual(keys(actions), [("ADVANCE", V, None)])

    def test_focus_actions_and_order(self):
        state = LearnerState(attempted_question_ids=["REC_001", "REC_003"])
        actions = build_available_actions(state, self.qb, ctx(RE, RECURSION_INTRODUCED))
        types = [a.action_type for a in actions]
        # Thứ tự cố định: ASK, HINT, EXPLAIN, REVIEW, ADVANCE
        order = [ActionType.ASK, ActionType.HINT, ActionType.EXPLAIN,
                 ActionType.REVIEW_PREREQUISITE, ActionType.ADVANCE]
        self.assertEqual(types, sorted(types, key=order.index))
        asks = [k for k in keys(actions) if k[0] == "ASK"]
        self.assertEqual(asks, [("ASK", RE, "Easy"), ("ASK", RE, "Medium"), ("ASK", RE, "Hard")])
        reviews = {k[1] for k in keys(actions) if k[0] == "REVIEW_PREREQUISITE"}
        self.assertEqual(reviews, {FU, SC, CO})

    def test_ask_excludes_exhausted_difficulty(self):
        state = LearnerState(attempted_question_ids=["REC_001", "REC_002"])  # hết 2 câu Easy
        actions = build_available_actions(state, self.qb, ctx(RE, RECURSION_INTRODUCED))
        self.assertNotIn(("ASK", RE, "Easy"), keys(actions))
        self.assertIn(("ASK", RE, "Medium"), keys(actions))

    def test_hint_requires_attempt_on_focus(self):
        actions = build_available_actions(LearnerState(), self.qb, ctx(RE, RECURSION_INTRODUCED))
        self.assertNotIn("HINT", [k[0] for k in keys(actions)])
        self.assertIn(("EXPLAIN", RE, None), keys(actions))

    def test_advance_requires_introduced_prerequisites_and_excludes_ancestors(self):
        actions = build_available_actions(LearnerState(), self.qb, ctx(FU, (V, DT, EX, FU)))
        advance = {k[1] for k in keys(actions) if k[0] == "ADVANCE"}
        self.assertIn(SC, advance)            # tiên quyết duy nhất là Functions
        self.assertIn("Strings", advance)     # DT + EX đã giới thiệu
        self.assertNotIn(RE, advance)         # cần Conditionals, chưa giới thiệu
        self.assertFalse(advance & {V, DT, EX})  # tổ tiên của focus → dùng REVIEW, không ADVANCE

    def test_introduced_derived_from_attempts(self):
        self.assertEqual(introduced_from_attempts(self.qb, ["VAR_001", "TYP_002", "VAR_003"]), (V, DT))
        state = LearnerState(attempted_question_ids=["VAR_001", "TYP_001", "EXP_001"])
        advance = {k[1] for k in keys(build_available_actions(state, self.qb, ctx(EX))) if k[0] == "ADVANCE"}
        self.assertIn(BO, advance)

    def test_unknown_focus_raises(self):
        with self.assertRaises(ValueError):
            build_available_actions(LearnerState(), self.qb, ctx("Pandas"))

    def test_alias_focus_is_canonicalized(self):
        actions = build_available_actions(LearnerState(), self.qb, ctx("Functions (def, params, return)", (V, DT, EX)))
        self.assertIn(("ASK", FU, "Easy"), keys(actions))


if __name__ == "__main__":
    unittest.main()
