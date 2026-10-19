"""Tests cho LLMPolicy và OpenAICompatibleClient, dùng model giả (không gọi mạng)."""
import json
import os
import tempfile
import unittest

from _helpers import RE, RECURSION_INTRODUCED, ctx, question_bank

from adaptive_learning.action_space import build_available_actions
from adaptive_learning.policies import BasePolicy, LLMDecisionError, LLMPolicy, RuleBasedPolicy
from adaptive_learning.policies.llm_clients import LLMClientError, OpenAICompatibleClient, load_env_file
from adaptive_learning.policies.llm_policy import build_user_prompt, parse_decision
from adaptive_learning.schemas import ActionType, LearnerState


class ScriptedModel:
    """Model giả trả lần lượt các output đã định sẵn và ghi lại messages nhận được."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class TestLLMPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qb = question_bank()
        cls.state = LearnerState(
            mastery={RE: 0.3, "Functions": 0.5},
            attempted_question_ids=["REC_001", "REC_003"],
            recent_errors=["REC_003: Tracing recursive calls / combining return values"],
            recent_actions=["ASK Recursion Medium -> wrong (REC_003)"],
        )
        cls.context = ctx(RE, RECURSION_INTRODUCED, last=("ASK", RE, "Medium"), last_qid="REC_003", last_correct=False)
        cls.actions = build_available_actions(cls.state, cls.qb, cls.context)
        cls.explain_index = next(i for i, a in enumerate(cls.actions) if a.action_type == ActionType.EXPLAIN)

    def policy(self, model, **kw):
        return LLMPolicy(model, concept_graph=self.qb.concept_graph, **kw)

    def test_valid_json_selects_indexed_action(self):
        model = ScriptedModel(json.dumps({"action_index": self.explain_index, "rationale": "learner is confused"}))
        decision = self.policy(model).select_action(self.state, self.actions, self.context)
        self.assertEqual(decision.action_type, ActionType.EXPLAIN)
        self.assertEqual(decision.concept, RE)
        self.assertEqual(decision.rationale, "learner is confused")
        self.assertTrue(BasePolicy.validate(decision, self.actions))
        self.assertEqual(len(model.calls), 1)

    def test_prompt_contains_state_and_numbered_actions_but_no_answers(self):
        prompt = build_user_prompt(self.state, self.actions, self.context, self.qb.concept_graph)
        self.assertIn("Recursion: 0.30", prompt)
        self.assertIn("Scope: not assessed", prompt)
        self.assertIn("Focus concept: Recursion", prompt)
        self.assertIn("REC_003 on Recursion -> wrong", prompt)
        self.assertIn(f"{self.explain_index}. EXPLAIN concept=Recursion", prompt)
        # Không lộ nội dung câu hỏi, đáp án hay ngưỡng của rule-based
        rec3 = self.qb["REC_003"]
        for leaked in (rec3.question.split("\n")[0], rec3.explanation, "correct_answer", "0.85", "0.75"):
            self.assertNotIn(leaked, prompt)

    def test_parses_fenced_json_and_string_index(self):
        action, rationale = parse_decision('Sure:\n```json\n{"action_index": "0", "rationale": "x"}\n```', self.actions)
        self.assertEqual(action, self.actions[0])
        self.assertEqual(rationale, "x")

    def test_parse_errors(self):
        for raw in ("", "no json here", '{"rationale": "x"}', '{"action_index": 99}',
                    '{"action_index": -1}', '{"action_index": true}', '{"action_index": 1.5}', "{bad json}"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_decision(raw, self.actions)

    def test_retry_after_invalid_output(self):
        model = ScriptedModel("I think EXPLAIN is best", json.dumps({"action_index": self.explain_index}))
        policy = self.policy(model, max_retries=2)
        decision = policy.select_action(self.state, self.actions, self.context)
        self.assertEqual(decision.action_type, ActionType.EXPLAIN)
        self.assertEqual(len(model.calls), 2)
        # Lần hỏi lại có kèm output sai và thông báo lỗi
        self.assertEqual(model.calls[1][-2]["role"], "assistant")
        self.assertIn("invalid", model.calls[1][-1]["content"])
        self.assertEqual([a["error"] is None for a in policy.last_trace["attempts"]], [False, True])

    def test_client_exception_is_retried(self):
        model = ScriptedModel(TimeoutError("slow"), json.dumps({"action_index": 0}))
        decision = self.policy(model).select_action(self.state, self.actions, self.context)
        self.assertEqual(decision.action_type, self.actions[0].action_type)

    def test_fallback_after_all_retries_fail(self):
        model = ScriptedModel("bad", "still bad")
        policy = self.policy(model, max_retries=1, fallback_policy=RuleBasedPolicy(concept_graph=self.qb.concept_graph))
        decision = policy.select_action(self.state, self.actions, self.context)
        self.assertTrue(decision.rationale.startswith("[fallback:rule_based]"))
        self.assertTrue(policy.last_trace["fallback_used"])
        self.assertTrue(BasePolicy.validate(decision, self.actions))

    def test_raises_without_fallback(self):
        policy = self.policy(ScriptedModel("bad"), max_retries=0)
        with self.assertRaises(LLMDecisionError):
            policy.select_action(self.state, self.actions, self.context)

    def test_constructor_validation(self):
        with self.assertRaises(TypeError):
            LLMPolicy("not callable")
        with self.assertRaises(ValueError):
            LLMPolicy(lambda m: "", max_retries=-1)


class TestOpenAICompatibleClient(unittest.TestCase):
    def test_request_format(self):
        client = OpenAICompatibleClient("https://example.test/v1/", "model-x", api_key="k", temperature=0.0)
        req = client.build_request([{"role": "user", "content": "hi"}])
        self.assertEqual(req.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(req.get_header("Authorization"), "Bearer k")
        body = json.loads(req.data)
        self.assertEqual(body["model"], "model-x")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(body["temperature"], 0.0)

    def test_call_extracts_content(self):
        client = OpenAICompatibleClient("http://local/v1", "m")
        client._send = lambda req: {"choices": [{"message": {"content": '{"action_index": 0}'}}]}
        self.assertEqual(client([{"role": "user", "content": "x"}]), '{"action_index": 0}')
        client._send = lambda req: {"error": "boom"}
        with self.assertRaises(LLMClientError):
            client([{"role": "user", "content": "x"}])

    def test_from_env(self):
        client = OpenAICompatibleClient.from_env({"LLM_BASE_URL": "http://local/v1", "LLM_MODEL": "m"})
        self.assertIsNone(client.api_key)
        with self.assertRaises(LLMClientError):
            OpenAICompatibleClient.from_env({"LLM_MODEL": "m"})

    def test_load_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# comment\nTEST_LLM_X='abc'\n\nTEST_LLM_Y=1\n")
            os.environ.pop("TEST_LLM_X", None)
            os.environ["TEST_LLM_Y"] = "keep"
            try:
                loaded = load_env_file(path)
                self.assertEqual(loaded, {"TEST_LLM_X": "abc", "TEST_LLM_Y": "1"})
                self.assertEqual(os.environ["TEST_LLM_X"], "abc")
                self.assertEqual(os.environ["TEST_LLM_Y"], "keep")
            finally:
                os.environ.pop("TEST_LLM_X", None)
                os.environ.pop("TEST_LLM_Y", None)


if __name__ == "__main__":
    unittest.main()
