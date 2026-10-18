"""Unit tests cho Schemas, QuestionBank và ConceptGraph (Thành viên 1)."""
import sys
import unittest
from pathlib import Path

# Thêm src vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_learning.schemas import (
    ALLOWED_DIFFICULTIES,
    Action,
    ActionType,
    Difficulty,
    LearnerState,
    PolicyDecision,
    Question,
    TurnResult,
)
from adaptive_learning.question_bank import ConceptGraph, QuestionBank


class TestSchemas(unittest.TestCase):
    def test_learner_state_valid(self):
        state = LearnerState(
            mastery={"Variables & assignment": 0.75, "Functions": 1.0, "Recursion": 0.0},
            recent_errors=["confused return with print"],
            recent_actions=["ASK:VAR_001"],
            attempted_question_ids=["VAR_001"],
        )
        self.assertEqual(state.mastery["Variables & assignment"], 0.75)
        self.assertEqual(state.mastery["Functions"], 1.0)
        self.assertEqual(state.mastery["Recursion"], 0.0)
        self.assertIsInstance(state.recent_errors, tuple)
        self.assertIsInstance(state.attempted_question_ids, tuple)
        self.assertEqual(state.attempted_question_ids, ("VAR_001",))

    def test_learner_state_mastery_bounds(self):
        # > 1.0 phải báo lỗi
        with self.assertRaises(ValueError):
            LearnerState(mastery={"Variables & assignment": 1.1})

        # < 0.0 phải báo lỗi
        with self.assertRaises(ValueError):
            LearnerState(mastery={"Variables & assignment": -0.05})

        # sai kiểu phải báo TypeError
        with self.assertRaises(TypeError):
            LearnerState(mastery={"Variables & assignment": "high"})

    def test_learner_state_serialization(self):
        state = LearnerState(
            mastery={"Variables & assignment": 0.8},
            recent_errors=["err1"],
            recent_actions=["act1"],
            attempted_question_ids=["VAR_001", "VAR_002"],
        )
        data = state.to_dict()
        self.assertIsInstance(data["recent_errors"], list)
        self.assertIsInstance(data["attempted_question_ids"], list)

        restored = LearnerState.from_dict(data)
        self.assertEqual(state, restored)

    def test_action_and_policy_decision(self):
        act = Action(
            action_type=ActionType.ASK,
            concept="Variables & assignment",
            difficulty="Easy",
            rationale="Test rationale",
        )
        self.assertEqual(act.action_type, ActionType.ASK)
        self.assertEqual(act.difficulty, "Easy")

        # Chuỗi chữ hoa được ép kiểu sang ActionType
        act2 = Action(action_type="ASK", difficulty=Difficulty.HARD)
        self.assertEqual(act2.action_type, ActionType.ASK)
        self.assertEqual(act2.difficulty, "Hard")

        # Sai difficulty phải báo ValueError
        with self.assertRaises(ValueError):
            Action(action_type=ActionType.ASK, difficulty="SuperHard")

        # Sai action_type phải báo ValueError
        with self.assertRaises(ValueError):
            Action(action_type="INVALID_ACTION")

        # PolicyDecision round-trip
        decision = PolicyDecision(action_type=ActionType.ADVANCE, concept="Data types")
        d_dict = decision.to_dict()
        restored_d = PolicyDecision.from_dict(d_dict)
        self.assertEqual(decision, restored_d)

    def test_question_schema(self):
        q = Question(
            question_id="VAR_001",
            concept="Variables & assignment",
            prerequisite_concept="None",
            question="What is b?",
            option_A="25",
            option_B="10",
            option_C="35",
            option_D="NameError",
            difficulty="Easy",
            correct_answer="B",
            common_error_type="Assignment misconception",
            explanation="b retains 10",
        )
        self.assertEqual(q.question_id, "VAR_001")
        self.assertEqual(q.options["B"], "10")
        self.assertEqual(q.prerequisites, ())

        # Tách nhiều prerequisite
        q2 = Question(
            question_id="REC_001",
            concept="Recursion",
            prerequisite_concept="Functions; Scope; Conditionals",
            question="Which is base case?",
            option_A="1",
            option_B="2",
            option_C="3",
            option_D="4",
            difficulty="Medium",
            correct_answer="C",
        )
        self.assertEqual(q2.prerequisites, ("Functions", "Scope", "Conditionals"))

        # Sai correct_answer phải báo ValueError
        with self.assertRaises(ValueError):
            Question(
                question_id="Q_ERR",
                concept="Functions",
                prerequisite_concept="None",
                question="q",
                option_A="1",
                option_B="2",
                option_C="3",
                option_D="4",
                difficulty="Easy",
                correct_answer="E",
            )


class TestConceptGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        graph_path = PROJECT_ROOT / "data" / "concept_graph.json"
        cls.graph = ConceptGraph.from_file(graph_path)

    def test_graph_loaded_correctly(self):
        self.assertEqual(len(self.graph.all_concepts()), 14)
        self.assertTrue(self.graph.is_dag())

    def test_canonicalize_aliases(self):
        # Trực tiếp
        self.assertEqual(self.graph.canonicalize("Variables & assignment"), "Variables & assignment")
        # Alias Functions
        self.assertEqual(self.graph.canonicalize("Functions (def, params, return)"), "Functions")
        self.assertEqual(self.graph.canonicalize("Functions (def/params/return)"), "Functions")
        # Alias Data type
        self.assertEqual(self.graph.canonicalize("Data type"), "Data types")

    def test_prerequisites_and_dependents(self):
        # Root node
        self.assertEqual(self.graph.get_prerequisites("Variables & assignment"), ())
        # Data types depends on Variables & assignment
        self.assertIn("Variables & assignment", self.graph.get_prerequisites("Data types"))
        # Recursion prerequisites
        rec_prereqs = self.graph.get_prerequisites("Recursion")
        self.assertIn("Functions", rec_prereqs)
        self.assertIn("Scope", rec_prereqs)
        self.assertIn("Conditionals", rec_prereqs)

    def test_topological_sort(self):
        order = self.graph.topological_sort()
        self.assertEqual(len(order), 14)
        self.assertEqual(order[0], "Variables & assignment")
        # Kiểm tra thứ tự: prerequisite phải đứng trước
        pos = {name: i for i, name in enumerate(order)}
        self.assertLess(pos["Variables & assignment"], pos["Functions"])
        self.assertLess(pos["Functions"], pos["Scope"])
        self.assertLess(pos["Scope"], pos["Recursion"])


class TestQuestionBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        csv_path = PROJECT_ROOT / "data" / "question_bank.csv"
        graph_path = PROJECT_ROOT / "data" / "concept_graph.json"
        cls.qb = QuestionBank(csv_path=csv_path, concept_graph=graph_path)

    def test_total_questions_and_concepts(self):
        self.assertEqual(len(self.qb), 70)
        concepts = self.qb.get_all_concepts()
        self.assertEqual(len(concepts), 14)

    def test_difficulty_distribution_per_concept(self):
        summary = self.qb.concept_summary()
        for concept, counts in summary.items():
            self.assertEqual(counts["Total"], 5, f"Concept '{concept}' must have 5 questions")
            self.assertEqual(counts["Easy"], 2, f"Concept '{concept}' must have 2 Easy questions")
            self.assertEqual(counts["Medium"], 2, f"Concept '{concept}' must have 2 Medium questions")
            self.assertEqual(counts["Hard"], 1, f"Concept '{concept}' must have 1 Hard question")

    def test_get_question_by_id(self):
        q = self.qb.get_question("VAR_001")
        self.assertIsNotNone(q)
        self.assertEqual(q.question_id, "VAR_001")
        self.assertEqual(q.concept, "Variables & assignment")
        self.assertEqual(q.correct_answer, "B")
        self.assertEqual(q.difficulty, "Easy")

        # Câu hỏi có ký tự tab trong raw CSV phải được strip chuẩn
        q4 = self.qb.get_question("VAR_004")
        self.assertIsNotNone(q4)
        self.assertEqual(q4.correct_answer, "A")

    def test_filter_by_concept_and_difficulty(self):
        # Lọc theo concept
        vars_qs = self.qb.get_questions(concept="Variables & assignment")
        self.assertEqual(len(vars_qs), 5)

        # Lọc theo concept + difficulty
        vars_easy = self.qb.get_questions(concept="Variables & assignment", difficulty="Easy")
        self.assertEqual(len(vars_easy), 2)
        for q in vars_easy:
            self.assertEqual(q.difficulty, "Easy")

        # Lọc bằng alias của concept
        func_qs = self.qb.get_questions(concept="Functions (def, params, return)")
        self.assertEqual(len(func_qs), 5)

    def test_exclude_attempted_questions(self):
        all_easy_vars = self.qb.get_questions(concept="Variables & assignment", difficulty="Easy")
        self.assertEqual(len(all_easy_vars), 2)

        # Loại bỏ 1 câu
        excluded = [all_easy_vars[0].question_id]
        remaining = self.qb.get_questions(
            concept="Variables & assignment", difficulty="Easy", exclude_ids=excluded
        )
        self.assertEqual(len(remaining), 1)
        self.assertNotIn(excluded[0], [q.question_id for q in remaining])

        # Loại bỏ cả 2 câu
        excluded_all = [q.question_id for q in all_easy_vars]
        none_remaining = self.qb.get_questions(
            concept="Variables & assignment", difficulty="Easy", exclude_ids=excluded_all
        )
        self.assertEqual(len(none_remaining), 0)

    def test_get_next_question(self):
        first_q = self.qb.get_next_question(concept="Data types", difficulty="Easy")
        self.assertIsNotNone(first_q)
        self.assertEqual(first_q.concept, "Data types")
        self.assertEqual(first_q.difficulty, "Easy")

        # Bỏ câu đầu tiên
        second_q = self.qb.get_next_question(
            concept="Data types", difficulty="Easy", exclude_ids=[first_q.question_id]
        )
        self.assertIsNotNone(second_q)
        self.assertNotEqual(first_q.question_id, second_q.question_id)


if __name__ == "__main__":
    unittest.main()
