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

from adaptive_learning.action_executor import ActionExecutor as RealExecutor
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


class TestEngineStopsAndExecutorRejections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qb = question_bank()

    def test_executor_rejection_keeps_state_and_context(self):
        class RejectingExecutor:
            def execute(self, decision, learner_state, respond=None):
                return TurnResult(decision=decision, valid=False, updated_state=None, error="no question")

        engine = LearningEngine(RuleBasedPolicy(concept_graph=self.qb.concept_graph), self.qb, RejectingExecutor())
        state = LearnerState()
        result = engine.run_turn(state)
        self.assertFalse(result.valid)
        self.assertIs(result.updated_state, state)
        self.assertEqual(engine.context, DecisionContext())

    def test_stalled_session_stops(self):
        explain = PolicyDecision(ActionType.EXPLAIN, concept="Functions")
        engine = LearningEngine(FixedPolicy(explain), self.qb, FakeExecutor(self.qb))
        engine.reset(DecisionContext(focus_concept="Functions", introduced_concepts=(V, "Functions")))
        session = engine.run_session(LearnerState(), max_steps=50, max_turns_without_question=4)
        self.assertEqual(session.stop_reason, "stalled")
        self.assertEqual(len(session.logs), 4)

    def test_questions_exhausted_stops(self):
        all_ids = tuple(q.question_id for q in self.qb.get_all_questions())
        engine = LearningEngine(RandomPolicy(seed=0), self.qb, FakeExecutor(self.qb))
        session = engine.run_session(LearnerState(attempted_question_ids=all_ids))
        self.assertEqual(session.stop_reason, "questions_exhausted")
        self.assertEqual(session.logs, ())

    def test_on_turn_callback_sees_every_turn(self):
        seen = []
        engine = LearningEngine(RuleBasedPolicy(concept_graph=self.qb.concept_graph), self.qb, FakeExecutor(self.qb))
        engine.run_session(LearnerState(), respond=lambda q: q.correct_answer, max_steps=6, on_turn=seen.append)
        self.assertEqual(len(seen), 6)


class TestIntegrationWithRealExecutor(unittest.TestCase):
    """Engine + ActionExecutor/StudentModel thật của Thành viên 2."""

    @classmethod
    def setUpClass(cls):
        cls.qb = question_bank()

    def test_rule_based_with_always_correct_learner_reaches_goal(self):
        engine = LearningEngine(RuleBasedPolicy(concept_graph=self.qb.concept_graph), self.qb, RealExecutor(self.qb))
        session = engine.run_session(LearnerState(), respond=lambda q: q.correct_answer, max_steps=200)
        self.assertEqual(session.stop_reason, "goal_reached")
        self.assertTrue(all(log.result.valid for log in session.logs))

    def test_rule_based_no_review_advance_ping_pong(self):
        import random
        rng = random.Random(0)
        engine = LearningEngine(RuleBasedPolicy(concept_graph=self.qb.concept_graph), self.qb, RealExecutor(self.qb))
        session = engine.run_session(
            LearnerState(), respond=lambda q: q.correct_answer if rng.random() < 0.6 else "Z", max_steps=400
        )
        self.assertIn(session.stop_reason, {"goal_reached", "questions_exhausted"})
        # Không có chuỗi dài lượt không hỏi câu nào
        streak = longest = 0
        for log in session.logs:
            streak = 0 if log.result.question_id else streak + 1
            longest = max(longest, streak)
        self.assertLessEqual(longest, 4)

    def test_every_policy_turn_valid_with_real_executor(self):
        engine = LearningEngine(RandomPolicy(seed=5), self.qb, RealExecutor(self.qb))
        session = engine.run_session(LearnerState(), respond=lambda q: "A", max_steps=150)
        self.assertTrue(all(log.result.valid for log in session.logs))


if __name__ == "__main__":
    unittest.main()
