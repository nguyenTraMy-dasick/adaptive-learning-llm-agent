"""Tests cho RuleBasedPolicy: mỗi luật R0–R6 (Thành viên 3)."""
import unittest

from _helpers import BO, CO, DT, EX, FU, RE, RECURSION_INTRODUCED, SC, V, ctx, question_bank

from adaptive_learning.action_space import build_available_actions
from adaptive_learning.policies import BasePolicy, RuleBasedPolicy, RuleConfig
from adaptive_learning.schemas import ActionType, LearnerState


class TestRuleBasedPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qb = question_bank()
        cls.policy = RuleBasedPolicy(concept_graph=cls.qb.concept_graph)

    def decide(self, mastery, attempted, context):
        state = LearnerState(mastery=mastery, attempted_question_ids=attempted)
        actions = build_available_actions(state, self.qb, context)
        decision = self.policy.select_action(state, actions, context)
        self.assertTrue(BasePolicy.validate(decision, actions), decision)
        return decision

    def assertDecision(self, decision, action_type, concept=None, difficulty=None, rule=None):
        self.assertEqual(decision.action_type, action_type, decision.rationale)
        self.assertEqual(decision.concept, concept, decision.rationale)
        self.assertEqual(decision.difficulty, difficulty, decision.rationale)
        if rule:
            self.assertTrue(decision.rationale.startswith(rule), decision.rationale)

    def test_r0_session_start(self):
        self.assertDecision(self.decide({}, [], ctx()), ActionType.ADVANCE, V, rule="R0")

    def test_r1_advance_when_mastered_prefers_unmastered_dependent(self):
        d = self.decide({V: 0.9, DT: 0.9, EX: 0.9, FU: 0.9}, ["FUN_001", "FUN_003"],
                        ctx(FU, (V, DT, EX, FU), last=("ASK", FU, "Medium"), last_qid="FUN_003", last_correct=True))
        self.assertDecision(d, ActionType.ADVANCE, SC, rule="R1")

    def test_r2_review_weakest_prerequisite(self):
        mastery = {FU: 0.45, SC: 0.55, CO: 0.9, RE: 0.25}
        d = self.decide(mastery, ["REC_001", "REC_003"],
                        ctx(RE, RECURSION_INTRODUCED, last=("ASK", RE, "Medium"), last_qid="REC_003", last_correct=False))
        self.assertDecision(d, ActionType.REVIEW_PREREQUISITE, FU, rule="R2")

    def test_r3_explain_when_prerequisites_strong(self):
        mastery = {FU: 0.9, SC: 0.9, CO: 0.9, RE: 0.25}
        d = self.decide(mastery, ["REC_001"],
                        ctx(RE, RECURSION_INTRODUCED, last=("ASK", RE, "Easy"), last_qid="REC_001", last_correct=False))
        self.assertDecision(d, ActionType.EXPLAIN, RE, rule="R3")

    def test_r3_not_repeated_after_explain(self):
        mastery = {FU: 0.9, SC: 0.9, CO: 0.9, RE: 0.25}
        d = self.decide(mastery, ["REC_001"],
                        ctx(RE, RECURSION_INTRODUCED, last=("EXPLAIN", RE, None), last_qid="REC_001", last_correct=False))
        self.assertDecision(d, ActionType.ASK, RE, "Easy", rule="R5")

    def test_r4_hint_on_medium_mastery(self):
        d = self.decide({RE: 0.6, FU: 0.9, SC: 0.9, CO: 0.9}, ["REC_003"],
                        ctx(RE, RECURSION_INTRODUCED, last=("ASK", RE, "Medium"), last_qid="REC_003", last_correct=False))
        self.assertDecision(d, ActionType.HINT, RE, rule="R4")

    def test_r5_difficulty_follows_mastery(self):
        context = ctx(RE, RECURSION_INTRODUCED, last=("ASK", RE, "Easy"), last_qid="REC_001", last_correct=True)
        for m, expected in ((0.2, "Easy"), (0.5, "Medium"), (0.8, "Hard")):
            with self.subTest(mastery=m):
                d = self.decide({RE: m}, ["REC_001"], context)
                self.assertDecision(d, ActionType.ASK, RE, expected, rule="R5")

    def test_r5_nearest_difficulty_when_target_exhausted(self):
        # Hết câu Hard (REC_005) → mastery 0.8 lấy Medium
        context = ctx(RE, RECURSION_INTRODUCED, last=("ASK", RE, "Hard"), last_qid="REC_005", last_correct=True)
        d = self.decide({RE: 0.8}, ["REC_005"], context)
        self.assertDecision(d, ActionType.ASK, RE, "Medium", rule="R5")

    def test_unknown_mastery_treated_as_configured_default(self):
        d = self.decide({}, [], ctx(FU, (V, DT, EX, FU), last=("ADVANCE", FU, None)))
        self.assertDecision(d, ActionType.ASK, FU, "Easy", rule="R5")
        policy = RuleBasedPolicy(RuleConfig(unknown_mastery=0.5), concept_graph=self.qb.concept_graph)
        state = LearnerState()
        context = ctx(FU, (V, DT, EX, FU))
        actions = build_available_actions(state, self.qb, context)
        self.assertEqual(policy.select_action(state, actions, context).difficulty, "Medium")

    def test_r6_focus_exhausted_moves_on(self):
        attempted = [f"FUN_00{i}" for i in range(1, 6)]
        d = self.decide({V: 0.9, DT: 0.9, EX: 0.9, FU: 0.6}, attempted,
                        ctx(FU, (V, DT, EX, FU), last=("ASK", FU, "Hard"), last_qid="FUN_005", last_correct=True))
        self.assertEqual(d.action_type, ActionType.ADVANCE)
        self.assertTrue(d.rationale.startswith("R6"), d.rationale)

    def test_does_not_mutate_inputs(self):
        state = LearnerState(mastery={RE: 0.2}, attempted_question_ids=["REC_001"])
        context = ctx(RE, RECURSION_INTRODUCED, last=("ASK", RE, "Easy"), last_qid="REC_001", last_correct=False)
        actions = build_available_actions(state, self.qb, context)
        before = (state.to_dict(), context.to_dict(), [a.to_dict() for a in actions])
        self.policy.select_action(state, actions, context)
        self.assertEqual(before, (state.to_dict(), context.to_dict(), [a.to_dict() for a in actions]))

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            RuleConfig(low=0.8, high=0.5)
        with self.assertRaises(ValueError):
            RuleConfig(advance=1.5)

    def test_works_without_graph(self):
        policy = RuleBasedPolicy()
        state = LearnerState(mastery={BO: 0.9})
        context = ctx(BO, (V, DT, EX, BO))
        actions = build_available_actions(state, self.qb, context)
        self.assertEqual(policy.select_action(state, actions, context).action_type, ActionType.ADVANCE)


if __name__ == "__main__":
    unittest.main()
