"""Tests cho learner mô phỏng và run_learning_experiment.py (Phase 6, Thành viên 3)."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from _helpers import PROJECT_ROOT, question_bank

sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from simulated_learner import LearnerProfile, SimulatedLearner, load_profiles  # noqa: E402

from adaptive_learning.schemas import ActionType, PolicyDecision, TurnResult  # noqa: E402


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_learning_experiment", PROJECT_ROOT / "evaluation" / "run_learning_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSimulatedLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qb = question_bank()

    def learner(self, **profile_kw):
        return SimulatedLearner(LearnerProfile(id="t", **profile_kw), self.qb, seed=0)

    def test_profiles_file_is_valid(self):
        profiles = load_profiles(PROJECT_ROOT / "data" / "simulated_learners.json", self.qb.concept_graph)
        self.assertGreaterEqual(len(profiles), 3)

    def test_invalid_profile(self):
        with self.assertRaises(ValueError):
            LearnerProfile(id="x", guess=0.6, slip=0.5)
        with self.assertRaises(ValueError):
            LearnerProfile(id="x", initial_knowledge={"Lists": 1.5})
        with self.assertRaises(ValueError):
            LearnerProfile.from_dict({"id": "x", "speed": 2})

    def test_p_correct_bounds_and_monotonic(self):
        low, high = self.learner(default_knowledge=0.0), self.learner(default_knowledge=1.0)
        q = self.qb["LIST_003"]
        self.assertAlmostEqual(low.p_correct(q), 0.25)
        self.assertAlmostEqual(high.p_correct(q), 0.90)
        mid = self.learner(default_knowledge=0.5)
        self.assertGreater(mid.p_correct(self.qb["LIST_001"]), mid.p_correct(self.qb["LIST_005"]))  # Easy > Hard

    def test_weak_prerequisite_lowers_performance(self):
        strong = self.learner(default_knowledge=0.8)
        weak = self.learner(default_knowledge=0.8, initial_knowledge={"Functions": 0.0})
        q = self.qb["SCO_003"]  # Scope phụ thuộc Functions
        self.assertLess(weak.p_correct(q), strong.p_correct(q))

    def test_observe_updates_knowledge_by_action(self):
        learner = self.learner(default_knowledge=0.2)
        before = dict(learner.knowledge)
        learner.observe(TurnResult(PolicyDecision(ActionType.EXPLAIN, "Lists"), valid=True))
        self.assertGreater(learner.knowledge["Lists"], before["Lists"])
        k = dict(learner.knowledge)
        learner.observe(TurnResult(PolicyDecision(ActionType.ADVANCE, "Lists"), valid=True))
        learner.observe(TurnResult(PolicyDecision(ActionType.EXPLAIN, "Lists"), valid=False))
        self.assertEqual(learner.knowledge, k)
        learner.observe(TurnResult(PolicyDecision(ActionType.ASK, "Lists", "Easy"), valid=True,
                                   question=self.qb["LIST_001"]))
        self.assertGreater(learner.knowledge["Lists"], k["Lists"])

    def test_respond_reproducible_and_letters(self):
        q = self.qb["VAR_001"]
        a = [SimulatedLearner(LearnerProfile(id="t"), self.qb, seed=3).respond(q) for _ in range(1)]
        b = [SimulatedLearner(LearnerProfile(id="t"), self.qb, seed=3).respond(q) for _ in range(1)]
        self.assertEqual(a, b)
        learner = self.learner(default_knowledge=0.5)
        self.assertTrue(all(learner.respond(q) in "ABCD" for _ in range(50)))

    def test_expected_test_score_range(self):
        self.assertAlmostEqual(self.learner(default_knowledge=0.0).expected_test_score(), 0.25)
        score = self.learner(default_knowledge=1.0).expected_test_score()
        self.assertGreater(score, 0.8)
        self.assertLessEqual(score, 0.9 + 1e-9)
        hard = self.learner(default_knowledge=1.0).p_correct(self.qb["LIST_005"])
        self.assertAlmostEqual(hard, 0.25 + 0.65 * 0.8)


class FirstActionModel:
    def __call__(self, messages):
        return '{"action_index": 0, "rationale": "test"}'


class TestLearningExperimentRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()
        cls.qb = question_bank()
        cls.config = cls.runner.load_config(cls.runner.DEFAULT_CONFIG)
        cls.profile = LearnerProfile(id="novice", default_knowledge=0.05)

    def session(self, name, seed=0, client=None, **cfg):
        config = {**self.config, "max_steps": 20, **cfg}
        policy = self.runner.make_policy(name, seed, self.qb, config, client)
        return self.runner.run_session(policy, self.profile, seed, self.qb, config)

    def test_session_record_fields(self):
        record = self.session("rule")
        for key in ("pre", "post", "gain", "n_questions", "turns", "goal_reached", "invalid_turns", "stop_reason"):
            self.assertIn(key, record)
        self.assertEqual(record["turns"], 20)
        self.assertGreaterEqual(record["gain"], 0.0)
        self.assertEqual(record["invalid_turns"], 0)
        json.dumps(record)

    def test_sessions_reproducible(self):
        self.assertEqual(self.session("random", seed=4), self.session("random", seed=4))
        self.assertEqual(self.session("rule", seed=4), self.session("rule", seed=4))

    def test_llm_policy_with_fake_client(self):
        record = self.session("llm", client=FirstActionModel())
        self.assertEqual(record["policy"], "llm")
        self.assertEqual(record["invalid_turns"], 0)

    def test_summarize(self):
        records = [self.session("random", seed=s) for s in range(3)] + [self.session("rule", seed=s) for s in range(3)]
        summary = self.runner.summarize(records)
        self.assertEqual(set(summary), {"random", "rule_based"})
        self.assertEqual(summary["random"]["gain"]["n"], 3)
        self.assertIn("novice", summary["rule_based"]["by_learner"])

    def test_main_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "exp.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = self.runner.main(["--policies", "random,rule", "--repeats", "1", "--max-steps", "8",
                                         "--learners", "novice", "--out", str(out),
                                         "--log-dir", str(Path(tmp) / "logs")])
            self.assertEqual(code, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(data["sessions"]), 2)
            self.assertTrue((Path(tmp) / "logs" / "rule" / "novice_seed0.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
