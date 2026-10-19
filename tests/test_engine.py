"""Tests cho LearningEngine với executor giả (Thành viên 3).

FakeExecutor chỉ mô phỏng interface ActionExecutor đề xuất cho Thành viên 2;
công thức mastery ở đây là giả, KHÔNG phải student model thật.
"""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from _helpers import V, question_bank

from adaptive_learning.engine import ActionExecutor, LearningEngine
from adaptive_learning.policies import BasePolicy, DecisionContext, RandomPolicy, RuleBasedPolicy
from adaptive_learning.schemas import ActionType, LearnerState, PolicyDecision, TurnResult


class FakeExecutor:
    """ASK: lấy câu đầu tiên còn lại, hỏi responder, chấm, cộng/trừ mastery cố định."""

    def __init__(self, qb):
        self.qb = qb
        self.calls = []

    def execute(self, decision, learner_state, respond=None):
        self.calls.append(decision)
        if decision.action_type != ActionType.ASK:
            return TurnResult(decision=decision, valid=True, feedback=decision.action_type.value,
                              updated_state=learner_state)
        q = self.qb.get_next_question(decision.concept, decision.difficulty,
                                      exclude_ids=learner_state.attempted_question_ids)
        answer = respond(q) if respond else q.correct_answer
        correct = answer == q.correct_answer
        m = learner_state.mastery.get(q.concept, 0.0)
        m = min(1.0, m + 0.3) if correct else max(0.0, m - 0.1)
        new_state = LearnerState(
            mastery={**learner_state.mastery, q.concept: m},
            recent_errors=learner_state.recent_errors + (() if correct else (q.common_error_type,)),
            recent_actions=learner_state.recent_actions + (f"ASK {q.question_id}",),
            attempted_question_ids=learner_state.attempted_question_ids + (q.question_id,),
        )
        return TurnResult(decision=decision, valid=True, question_id=q.question_id, question=q,
                          correct=correct, updated_state=new_state)


class FixedPolicy(BasePolicy):
    name = "fixed"

    def __init__(self, decision):
        self.decision = decision

    def select_action(self, learner_state, available_actions, context=None):
        return self.decision


class CrashingPolicy(BasePolicy):
    name = "crash"

    def select_action(self, learner_state, available_actions, context=None):
        raise RuntimeError("model unavailable")


class TestLearningEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qb = question_bank()

    def engine(self, policy):
        executor = FakeExecutor(self.qb)
        return LearningEngine(policy, self.qb, executor), executor

    def test_fake_executor_satisfies_protocol(self):
        self.assertIsInstance(FakeExecutor(self.qb), ActionExecutor)

    def test_first_turn_advances_and_sets_focus(self):
        engine, executor = self.engine(RuleBasedPolicy(concept_graph=self.qb.concept_graph))
        result = engine.run_turn(LearnerState())
        self.assertTrue(result.valid)
        self.assertEqual(result.decision.action_type, ActionType.ADVANCE)
        self.assertEqual(engine.context.focus_concept, V)
        self.assertEqual(engine.context.introduced_concepts, (V,))
        self.assertEqual(engine.context.step, 1)

    def test_ask_updates_state_and_last_question(self):
        engine, _ = self.engine(RuleBasedPolicy(concept_graph=self.qb.concept_graph))
        state = engine.run_turn(LearnerState()).updated_state
        result = engine.run_turn(state, respond=lambda q: "Z")  # trả lời sai
        self.assertEqual(result.decision.action_type, ActionType.ASK)
        self.assertEqual(engine.context.last_question_id, result.question_id)
        self.assertEqual(engine.context.last_question_concept, V)
        self.assertIs(engine.context.last_correct, False)
        self.assertIn(result.question_id, result.updated_state.attempted_question_ids)

    def test_invalid_decision_is_rejected_and_not_executed(self):
        bad = PolicyDecision(ActionType.ASK, concept="Recursion", difficulty="Hard")
        engine, executor = self.engine(FixedPolicy(bad))
        state = LearnerState()
        result = engine.run_turn(state)
        self.assertFalse(result.valid)
        self.assertEqual(result.error, "decision is not in available_actions")
        self.assertIs(result.updated_state, state)
        self.assertEqual(executor.calls, [])
        self.assertEqual(engine.context, DecisionContext())
        self.assertEqual(len(engine.logs), 1)

    def test_policy_exception_becomes_invalid_turn(self):
        engine, executor = self.engine(CrashingPolicy())
        result = engine.run_turn(LearnerState())
        self.assertFalse(result.valid)
        self.assertIn("model unavailable", result.error)
        self.assertEqual(executor.calls, [])

    def test_review_prerequisite_changes_focus(self):
        engine, _ = self.engine(FixedPolicy(PolicyDecision(ActionType.REVIEW_PREREQUISITE, concept="Functions")))
        engine.reset(DecisionContext(focus_concept="Scope", introduced_concepts=(V, "Data types",
                                     "Expressions & operators", "Functions", "Scope")))
        result = engine.run_turn(LearnerState())
        self.assertTrue(result.valid)
        self.assertEqual(engine.context.focus_concept, "Functions")

    def test_session_with_correct_learner_reaches_goal_or_stops(self):
        engine, _ = self.engine(RuleBasedPolicy(concept_graph=self.qb.concept_graph))
        session = engine.run_session(LearnerState(), respond=lambda q: q.correct_answer, max_steps=400)
        self.assertIn(session.stop_reason, {"goal_reached", "no_actions"})
        self.assertTrue(all(log.result.valid for log in session.logs))
        mastered = [c for c in self.qb.get_all_concepts() if session.final_state.mastery.get(c, 0) >= 0.85]
        self.assertEqual(len(mastered), len(self.qb.get_all_concepts()))

    def test_session_stops_after_consecutive_invalid(self):
        engine, _ = self.engine(CrashingPolicy())
        session = engine.run_session(LearnerState(), max_steps=50, max_consecutive_invalid=3)
        self.assertEqual(session.stop_reason, "too_many_invalid")
        self.assertEqual(len(session.logs), 3)

    def test_random_session_is_reproducible(self):
        def run():
            engine, _ = self.engine(RandomPolicy(seed=11))
            s = engine.run_session(LearnerState(), respond=lambda q: "A", max_steps=40)
            return [log.result.decision.to_dict() for log in s.logs]
        self.assertEqual(run(), run())

    def test_write_log_jsonl(self):
        engine, _ = self.engine(RuleBasedPolicy(concept_graph=self.qb.concept_graph))
        engine.run_session(LearnerState(), respond=lambda q: q.correct_answer, max_steps=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = engine.write_log(Path(tmp) / "logs" / "session.jsonl")
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 5)
        first = json.loads(lines[0])
        self.assertEqual(set(first), {"turn", "policy", "context", "state_before", "available_actions",
                                      "result", "policy_trace"})
        self.assertEqual(first["policy"], "rule_based")

    def test_executor_without_updated_state_keeps_state(self):
        class LazyExecutor(FakeExecutor):
            def execute(self, decision, learner_state, respond=None):
                return replace(super().execute(decision, learner_state, respond), updated_state=None)

        engine = LearningEngine(RuleBasedPolicy(concept_graph=self.qb.concept_graph), self.qb, LazyExecutor(self.qb))
        state = LearnerState()
        self.assertIs(engine.run_turn(state).updated_state, state)


if __name__ == "__main__":
    unittest.main()
