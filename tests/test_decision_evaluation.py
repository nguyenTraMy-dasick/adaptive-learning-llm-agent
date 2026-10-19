"""Tests cho evaluation/metrics.py và run_decision_evaluation.py (Thành viên 3)."""
import importlib.util
import json
import sys
import unittest

from _helpers import PROJECT_ROOT, question_bank

sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

import metrics  # noqa: E402

from adaptive_learning.policies import RandomPolicy, RuleBasedPolicy  # noqa: E402


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_decision_evaluation", PROJECT_ROOT / "evaluation" / "run_decision_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMetrics(unittest.TestCase):
    def test_label_agreement(self):
        label = {
            "preferred_actions": [{"action_type": "EXPLAIN", "concept": "Lists", "difficulty": None}],
            "acceptable_actions": [{"action_type": "ASK", "concept": "Lists", "difficulty": "Easy"}],
        }
        explain = {"action_type": "EXPLAIN", "concept": "Lists", "difficulty": None}
        ask_easy = {"action_type": "ASK", "concept": "Lists", "difficulty": "Easy"}
        explain_other = {"action_type": "EXPLAIN", "concept": "Strings", "difficulty": None}
        self.assertEqual(metrics.label_agreement(explain, label), {"exact": True, "acceptable": True, "type": True})
        self.assertEqual(metrics.label_agreement(ask_easy, label), {"exact": False, "acceptable": True, "type": False})
        self.assertEqual(metrics.label_agreement(explain_other, label)["type"], True)
        self.assertEqual(metrics.label_agreement(None, label)["acceptable"], False)

    def test_scenario_agreement_averages_annotators(self):
        a = {"action_type": "HINT", "concept": "X", "difficulty": None}
        labels = [{"preferred_actions": [a]}, {"preferred_actions": [{**a, "action_type": "EXPLAIN"}]}, {}]
        self.assertEqual(metrics.scenario_agreement(a, labels)["exact"], 0.5)
        self.assertIsNone(metrics.scenario_agreement(a, []))

    def test_auto_flags(self):
        ctx = {"focus_concept": "Lists", "last_action": {"action_type": "EXPLAIN", "concept": "Lists", "difficulty": None}}
        avail = [{"action_type": "ASK", "concept": "Lists", "difficulty": "Easy"}]
        flags = metrics.auto_flags({"action_type": "ADVANCE", "concept": "Dictionaries"}, {"Lists": 0.3}, ctx, avail)
        self.assertEqual(flags, ["premature_advance"])
        self.assertEqual(metrics.auto_flags({"action_type": "ASK", "concept": "Lists", "difficulty": "Hard"},
                                            {"Lists": 0.2}, ctx, avail), ["too_hard"])
        self.assertEqual(metrics.auto_flags({"action_type": "EXPLAIN", "concept": "Lists", "difficulty": None},
                                            {"Lists": 0.3}, ctx, avail), ["repeated_support"])
        self.assertEqual(metrics.auto_flags(None, {}, ctx, avail), ["no_decision"])

    def test_learning_metrics(self):
        self.assertAlmostEqual(metrics.learning_gain(0.4, 0.7), 0.3)
        self.assertAlmostEqual(metrics.normalized_gain(0.4, 0.7), 0.5)
        self.assertIsNone(metrics.normalized_gain(1.0, 1.0))
        self.assertAlmostEqual(metrics.efficiency(0.3, 10), 0.03)
        self.assertIsNone(metrics.efficiency(0.3, 0))


class TestDecisionEvaluationRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()
        cls.qb = question_bank()
        cls.scenarios = cls.runner.load_scenarios(cls.runner.DEFAULT_SCENARIOS)

    def test_scenarios_are_consistent(self):
        ids = [s["id"] for s in self.scenarios]
        self.assertEqual(len(ids), len(set(ids)))
        for sc in self.scenarios:
            with self.subTest(scenario=sc["id"]):
                for qid in sc["learner_state"]["attempted_question_ids"]:
                    self.assertTrue(self.qb.has_question(qid), qid)
                focus = sc["context"]["focus_concept"]
                if focus:
                    self.assertTrue(self.qb.concept_graph.has_concept(focus))

    def test_all_policies_valid_on_every_scenario(self):
        for policy in (RandomPolicy(seed=0), RuleBasedPolicy(concept_graph=self.qb.concept_graph)):
            records = self.runner.evaluate_policy(policy, self.scenarios, self.qb)
            self.assertEqual(len(records), len(self.scenarios))
            self.assertTrue(all(r["valid"] for r in records), policy.name)
            summary = metrics.summarize_decisions(records)
            self.assertEqual(summary["valid_rate"], 1.0)
            json.dumps(records)  # kết quả phải serialize được

    def test_agreement_computed_when_labels_exist(self):
        sc = json.loads(json.dumps(self.scenarios[0]))  # S01: chỉ có ADVANCE Variables
        sc["expert_labels"] = [{"annotator": "T", "preferred_actions": [
            {"action_type": "ADVANCE", "concept": "Variables & assignment", "difficulty": None}]}]
        records = self.runner.evaluate_policy(RuleBasedPolicy(), [sc], self.qb)
        self.assertEqual(records[0]["agreement"]["exact"], 1.0)
        self.assertEqual(metrics.summarize_decisions(records)["exact_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
