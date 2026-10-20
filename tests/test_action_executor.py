import unittest

from _helpers import question_bank
from adaptive_learning.action_executor import ActionExecutor
from adaptive_learning.schemas import ActionType, LearnerState, PolicyDecision


class TestActionExecutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qb = question_bank()
        cls.executor = ActionExecutor(cls.qb)

    def test_ask_selects_and_grades_from_question_bank(self):
        decision = PolicyDecision(ActionType.ASK, "Variables & assignment", "Easy")
        result = self.executor.execute(decision, LearnerState(), lambda q: "B")
        self.assertTrue(result.valid)
        self.assertEqual(result.question_id, "VAR_001")
        self.assertTrue(result.correct)
        self.assertIn("VAR_001", result.updated_state.attempted_question_ids)

    def test_wrong_answer_records_error(self):
        decision = PolicyDecision(ActionType.ASK, "Variables & assignment", "Easy")
        result = self.executor.execute(decision, LearnerState(), lambda q: "A")
        self.assertFalse(result.correct)
        self.assertTrue(result.updated_state.recent_errors)

    def test_repeated_question_is_excluded(self):
        decision = PolicyDecision(ActionType.ASK, "Variables & assignment", "Easy")
        result = self.executor.execute(
            decision, LearnerState(attempted_question_ids=("VAR_001",)), lambda q: q.correct_answer
        )
        self.assertTrue(result.valid)
        self.assertNotEqual(result.question_id, "VAR_001")

    def test_exhausted_question_is_explicitly_invalid(self):
        attempted = tuple(q.question_id for q in self.qb.get_questions("Variables & assignment", "Easy"))
        decision = PolicyDecision(ActionType.ASK, "Variables & assignment", "Easy")
        state = LearnerState(attempted_question_ids=attempted)
        result = self.executor.execute(decision, state)
        self.assertFalse(result.valid)
        self.assertIs(result.updated_state, state)
        self.assertIn("No unanswered question", result.error)

    def test_ask_without_responder_does_not_assume_correct_answer(self):
        decision = PolicyDecision(ActionType.ASK, "Variables & assignment", "Easy")
        state = LearnerState()
        result = self.executor.execute(decision, state)
        self.assertFalse(result.valid)
        self.assertIs(result.updated_state, state)
        self.assertIn("requires a responder", result.error)
        self.assertIsNone(result.correct)

    def test_support_action_preserves_mastery_and_records_history(self):
        state = LearnerState(mastery={"Variables & assignment": 0.4})
        result = self.executor.execute(PolicyDecision(ActionType.EXPLAIN, "Variables & assignment"), state)
        self.assertTrue(result.valid)
        self.assertEqual(result.updated_state.mastery, state.mastery)
        self.assertIn("EXPLAIN:Variables & assignment", result.updated_state.recent_actions)
        self.assertTrue(result.feedback)

    def test_malformed_ask_is_rejected(self):
        with self.assertRaises(ValueError):
            self.executor.execute(PolicyDecision(ActionType.ASK, "Variables & assignment"), LearnerState())
