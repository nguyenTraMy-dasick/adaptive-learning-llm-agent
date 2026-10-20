import unittest

from _helpers import question_bank
from adaptive_learning.schemas import LearnerState
from adaptive_learning.student_model import StudentModel


class TestStudentModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.question = question_bank().get_question("VAR_001")

    def test_grading_and_ema(self):
        model = StudentModel()
        self.assertTrue(model.grade_answer(self.question, " b "))
        self.assertFalse(model.grade_answer(self.question, "A"))
        self.assertAlmostEqual(model.update_mastery(0.0, True), 0.4)
        self.assertAlmostEqual(model.update_mastery(1.0, False), 0.6)

    def test_update_records_correct_answer(self):
        state, correct = StudentModel().update_state(LearnerState(), self.question, "B")
        self.assertTrue(correct)
        self.assertAlmostEqual(state.mastery["Variables & assignment"], 0.4)
        self.assertEqual(state.attempted_question_ids, ("VAR_001",))
        self.assertEqual(state.recent_errors, ())

    def test_update_records_wrong_answer_and_caps_history(self):
        state, correct = StudentModel(history_limit=1).update_state(LearnerState(), self.question, "A")
        self.assertFalse(correct)
        self.assertEqual(state.mastery["Variables & assignment"], 0.0)
        self.assertEqual(state.recent_errors, (self.question.common_error_type,))
        self.assertEqual(len(state.recent_actions), 1)

    def test_repeated_question_not_added_twice(self):
        state, _ = StudentModel().update_state(
            LearnerState(attempted_question_ids=("VAR_001",)), self.question, "B"
        )
        self.assertEqual(state.attempted_question_ids, ("VAR_001",))

    def test_mastery_threshold_reachable_within_one_concept(self):
        # Mỗi concept chỉ có 5 câu: ngưỡng 0.85 phải đạt được sau 4 câu đúng liên tiếp.
        model, mastery = StudentModel(), 0.0
        for _ in range(4):
            mastery = model.update_mastery(mastery, True)
        self.assertGreaterEqual(mastery, 0.85)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            StudentModel(learning_rate=0)
        with self.assertRaises(ValueError):
            StudentModel(history_limit=1.5)