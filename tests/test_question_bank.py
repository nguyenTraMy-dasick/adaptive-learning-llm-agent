"""Unit tests cho Schemas, QuestionBank và ConceptGraph (Thành viên 1)."""
import copy
import csv
import importlib.util
import pickle
import random
import sys
import tempfile
import textwrap
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

        # correct_answer luôn được strip/upper khi load (phòng ký tự tab như '\tA')
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


class TestLearnerStateImmutability(unittest.TestCase):
    def test_mastery_is_read_only(self):
        state = LearnerState(mastery={"Lists": 0.5})
        with self.assertRaises(TypeError):
            state.mastery["Lists"] = 5.0
        with self.assertRaises(TypeError):
            state.mastery.update({"Lists": 0.9})
        with self.assertRaises(TypeError):
            del state.mastery["Lists"]
        self.assertEqual(state.mastery, {"Lists": 0.5})

    def test_update_by_creating_new_state(self):
        state = LearnerState(mastery={"Lists": 0.5, "Recursion": 0.2})
        new_state = LearnerState(mastery={**state.mastery, "Lists": 0.6})
        self.assertEqual(new_state.mastery["Lists"], 0.6)
        self.assertEqual(state.mastery["Lists"], 0.5)

    def test_rejects_nan_inf_and_bool(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                LearnerState(mastery={"Lists": bad})
        with self.assertRaises(TypeError):
            LearnerState(mastery={"Lists": True})

    def test_copy_pickle_and_hash(self):
        state = LearnerState(mastery={"Lists": 0.5}, attempted_question_ids=["LIST_001"])
        self.assertEqual(copy.deepcopy(state), state)
        self.assertEqual(pickle.loads(pickle.dumps(state)), state)
        self.assertEqual(hash(state), hash(LearnerState(mastery={"Lists": 0.5}, attempted_question_ids=["LIST_001"])))


class TestQuestionBankDataIntegrity(unittest.TestCase):
    """Các ràng buộc dữ liệu mà student model / policy dựa vào."""

    @classmethod
    def setUpClass(cls):
        cls.graph = ConceptGraph.from_file(PROJECT_ROOT / "data" / "concept_graph.json")
        cls.qb = QuestionBank(
            csv_path=PROJECT_ROOT / "data" / "question_bank.csv",
            concept_graph=cls.graph,
        )

    def test_prerequisites_are_canonical_ancestors(self):
        for q in self.qb.get_all_questions():
            ancestors = self.graph.get_ancestors(q.concept)
            for p in q.prerequisites:
                with self.subTest(question=q.question_id, prerequisite=p):
                    self.assertTrue(self.graph.has_concept(p))
                    self.assertEqual(p, self.graph.canonicalize(p), "prerequisite phải là canonical name")
                    self.assertIn(p, ancestors, "prerequisite phải là tổ tiên của concept trong graph")

    def test_alias_prerequisites_are_canonicalized_on_load(self):
        self.assertEqual(self.qb["REC_004"].prerequisites, ("Functions", "Scope", "Conditionals"))
        self.assertEqual(self.qb["SCO_001"].prerequisites, ("Functions",))
        self.assertEqual(self.qb["VAR_001"].prerequisites, ())

    def test_options_are_distinct(self):
        for q in self.qb.get_all_questions():
            with self.subTest(question=q.question_id):
                self.assertEqual(len(set(q.options.values())), 4)

    def test_question_code_compiles(self):
        for q in self.qb.get_all_questions():
            if "\n" not in q.question:
                continue
            code = textwrap.dedent(q.question.split("\n", 1)[1]).strip("\n")
            with self.subTest(question=q.question_id):
                compile(code, q.question_id, "exec")

    def test_random_next_question_is_reproducible_with_seed(self):
        picks_1 = [self.qb.get_next_question(concept="Lists", strategy="random", rng=random.Random(7)).question_id
                   for _ in range(3)]
        picks_2 = [self.qb.get_next_question(concept="Lists", strategy="random", rng=random.Random(7)).question_id
                   for _ in range(3)]
        self.assertEqual(picks_1, picks_2)
        with self.assertRaises(ValueError):
            self.qb.get_next_question(concept="Lists", strategy="hardest")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(Question.__dataclass_fields__.keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "question_id": "LIST_900",
        "concept": "Lists",
        "prerequisite_concept": "Variables & assignment",
        "question": "What is printed?\nnums = [1, 2]\nprint(nums[0])",
        "option_A": "1", "option_B": "2", "option_C": "[1, 2]", "option_D": "IndexError",
        "difficulty": "Easy",
        "correct_answer": "A",
        "common_error_type": "Zero-based indexing",
        "option_error_mapping": "B -> counts from 1",
        "explanation": "Index 0 is the first element.",
    }
    row.update(overrides)
    return row


class TestValidatorDetectsDataErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "validate_question_bank", PROJECT_ROOT / "scripts" / "validate_question_bank.py"
        )
        cls.validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.validator)
        cls.graph_path = PROJECT_ROOT / "data" / "concept_graph.json"

    def _errors_for(self, rows: list[dict[str, str]]) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "qb.csv"
            _write_csv(csv_path, rows)
            report = self.validator.ValidationReport()
            graph_data = self.validator.validate_concept_graph(self.graph_path, report)
            self.validator.validate_question_bank_csv(csv_path, graph_data, report)
            return report.errors

    def test_valid_row_has_no_errors(self):
        self.assertEqual(self._errors_for([_row()]), [])

    def test_duplicate_options(self):
        errors = self._errors_for([_row(option_C="1")])
        self.assertTrue(any("duplicates" in e for e in errors), errors)

    def test_self_prerequisite(self):
        errors = self._errors_for([_row(prerequisite_concept="Lists")])
        self.assertTrue(any("own concept" in e for e in errors), errors)

    def test_prerequisite_not_ancestor(self):
        errors = self._errors_for([_row(prerequisite_concept="Recursion")])
        self.assertTrue(any("not an ancestor" in e for e in errors), errors)

    def test_broken_indentation(self):
        errors = self._errors_for([_row(question="What is printed?\ndef f(n):\n  if n == 0:\n  return 0")])
        self.assertTrue(any("does not compile" in e for e in errors), errors)


class TestQuestionBankLoadErrors(unittest.TestCase):
    def test_duplicate_question_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "qb.csv"
            _write_csv(csv_path, [_row(), _row()])
            with self.assertRaises(ValueError):
                QuestionBank(csv_path=csv_path, concept_graph=PROJECT_ROOT / "data" / "concept_graph.json")


if __name__ == "__main__":
    unittest.main()
