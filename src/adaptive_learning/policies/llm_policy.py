"""Thành viên 3: Policy C — LLM Agent. Model chỉ *đề xuất* action.

Luồng một lần ra quyết định:
1. Serialize learner state + DecisionContext + danh sách available_actions (đánh số)
   thành prompt. Không gửi câu hỏi, đáp án đúng hay giải thích của question bank.
2. Gọi `model_client(messages) -> str` (callable được inject, không gắn với nhà cung cấp).
3. Parse JSON `{"action_index": <số>, "rationale": "..."}`; LLM chọn theo *số thứ tự*
   nên không thể bịa concept hay difficulty ngoài danh sách.
4. Output sai định dạng hoặc chọn ngoài danh sách → gửi lại lỗi cho model, thử tối đa
   `max_retries` lần. Vẫn lỗi → dùng `fallback_policy` nếu có, nếu không raise
   LLMDecisionError (engine ghi nhận bước đó là invalid).

Model không chấm câu trả lời và không cập nhật mastery. Rationale chỉ để phân tích.
Prompt cố ý KHÔNG chứa ngưỡng mastery của rule-based policy, để LLM tự ra quyết định.
Toàn bộ quá trình gọi (prompt, output thô, lỗi parse, fallback) được lưu ở `last_trace`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Sequence

from adaptive_learning.policies.base import BasePolicy, DecisionContext, to_decision
from adaptive_learning.schemas import Action, LearnerState, PolicyDecision

Message = Mapping[str, str]
ModelClient = Callable[[Sequence[Message]], str]

SYSTEM_PROMPT = """You are the pedagogical decision module of an adaptive tutoring system for introductory Python.
At each step you choose exactly ONE next instructional action for the learner from a list of available actions.
You do not write questions, grade answers or update the learner model; other components do that.

Action types:
- ASK: give the learner a multiple-choice question on the concept at the given difficulty (Easy, Medium, Hard).
- HINT: give a hint about the most recent question on the focus concept.
- EXPLAIN: give an explanation of the focus concept.
- REVIEW_PREREQUISITE: switch to reviewing a prerequisite concept of the current focus concept.
- ADVANCE: move on to a new focus concept.

Goal: help the learner master all concepts efficiently, taking into account the prerequisite structure,
the learner's mastery estimates, recent mistakes and recent actions.

Reply with a single JSON object and nothing else:
{"action_index": <number of the chosen action>, "rationale": "<one or two sentences>"}"""


class LLMDecisionError(RuntimeError):
    """LLM không trả về action hợp lệ sau mọi lần thử và không có fallback."""


def _format_mastery(value: float | None) -> str:
    return "not assessed" if value is None else f"{value:.2f}"


def build_user_prompt(
    learner_state: LearnerState,
    available_actions: Sequence[Action],
    context: DecisionContext | None = None,
    concept_graph: Any = None,
    max_history: int = 5,
) -> str:
    """Tạo nội dung user prompt; tách riêng để test và để đưa vào báo cáo."""
    context = context or DecisionContext()
    lines: list[str] = []

    if concept_graph is not None:
        lines.append("Concepts in curriculum order (prerequisites in brackets):")
        for concept in concept_graph.topological_sort():
            prereqs = ", ".join(concept_graph.get_prerequisites(concept)) or "none"
            lines.append(f"- {concept} [{prereqs}]")
        lines.append("")
        concepts = list(concept_graph.topological_sort())
    else:
        concepts = list(learner_state.mastery.keys())

    lines.append("Learner mastery estimates (0 = not mastered, 1 = fully mastered):")
    for concept in concepts:
        lines.append(f"- {concept}: {_format_mastery(learner_state.mastery.get(concept))}")
    extra = [c for c in learner_state.mastery if c not in concepts]
    for concept in extra:
        lines.append(f"- {concept}: {_format_mastery(learner_state.mastery[concept])}")
    lines.append("")

    lines.append("Session context:")
    lines.append(f"- Focus concept: {context.focus_concept or 'none (session start)'}")
    lines.append(f"- Steps taken so far: {context.step}")
    if context.last_action is not None:
        la = context.last_action
        detail = " ".join(x for x in (la.concept, la.difficulty) if x)
        lines.append(f"- Previous action: {la.action_type.value} {detail}".rstrip())
    if context.last_question_id is not None:
        result = {True: "correct", False: "wrong", None: "not graded"}[context.last_correct]
        lines.append(
            f"- Most recent question: {context.last_question_id} on {context.last_question_concept} -> {result}"
        )
    lines.append(f"- Questions answered in total: {len(learner_state.attempted_question_ids)}")
    lines.append("")

    recent_errors = list(learner_state.recent_errors)[-max_history:]
    recent_actions = list(learner_state.recent_actions)[-max_history:]
    lines.append("Recent mistakes (oldest first):")
    lines.extend(f"- {e}" for e in recent_errors) if recent_errors else lines.append("- none")
    lines.append("Recent actions (oldest first):")
    lines.extend(f"- {a}" for a in recent_actions) if recent_actions else lines.append("- none")
    lines.append("")

    lines.append("Available actions (choose exactly one by its number):")
    for i, action in enumerate(available_actions):
        parts = [action.action_type.value]
        if action.concept:
            parts.append(f"concept={action.concept}")
        if action.difficulty:
            parts.append(f"difficulty={action.difficulty}")
        lines.append(f"{i}. " + " ".join(parts))
    return "\n".join(lines)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_decision(raw: str, available_actions: Sequence[Action]) -> tuple[Action, str]:
    """Parse output của model thành (action, rationale); raise ValueError nếu không hợp lệ.

    Chấp nhận JSON thuần hoặc JSON nằm trong khối ```json ... ```. Trường bắt buộc là
    `action_index` (int hoặc chuỗi số); `rationale` là tùy chọn.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty response")
    match = _JSON_OBJECT.search(raw)
    if not match:
        raise ValueError("no JSON object found in response")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict) or "action_index" not in data:
        raise ValueError('JSON must contain "action_index"')

    index = data["action_index"]
    if isinstance(index, bool):
        raise ValueError("action_index must be an integer")
    if isinstance(index, str) and index.strip().lstrip("-").isdigit():
        index = int(index.strip())
    if isinstance(index, float) and index.is_integer():
        index = int(index)
    if not isinstance(index, int):
        raise ValueError("action_index must be an integer")
    if not 0 <= index < len(available_actions):
        raise ValueError(f"action_index {index} is out of range 0..{len(available_actions) - 1}")

    rationale = data.get("rationale", "")
    return available_actions[index], rationale if isinstance(rationale, str) else json.dumps(rationale)


class LLMPolicy(BasePolicy):
    name = "llm"

    def __init__(
        self,
        model_client: ModelClient,
        concept_graph: Any = None,
        max_retries: int = 2,
        fallback_policy: BasePolicy | None = None,
        max_history: int = 5,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        """
        Args:
            model_client: callable nhận list message {"role", "content"} và trả về text.
                Xem `llm_clients.OpenAICompatibleClient`, hoặc truyền hàm giả khi test.
            concept_graph: ConceptGraph để đưa cấu trúc tiên quyết vào prompt (nên có).
            max_retries: số lần hỏi lại khi output không hợp lệ (tổng số lần gọi = 1 + max_retries).
            fallback_policy: policy dùng khi LLM vẫn lỗi; None thì raise LLMDecisionError.
            max_history: số mục recent_errors / recent_actions đưa vào prompt.
        """
        if not callable(model_client):
            raise TypeError("model_client must be callable: (messages) -> str")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.model_client = model_client
        self.concept_graph = concept_graph
        self.max_retries = max_retries
        self.fallback_policy = fallback_policy
        self.max_history = max_history
        self.system_prompt = system_prompt
        self.last_trace: dict[str, Any] = {}

    def select_action(
        self,
        learner_state: LearnerState,
        available_actions: tuple[Action, ...],
        context: DecisionContext | None = None,
    ) -> PolicyDecision:
        self._require_actions(available_actions)
        prompt = build_user_prompt(
            learner_state, available_actions, context, self.concept_graph, self.max_history
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        attempts: list[dict[str, Any]] = []
        self.last_trace = {"prompt": prompt, "attempts": attempts, "fallback_used": False}

        for _ in range(self.max_retries + 1):
            try:
                raw = self.model_client(list(messages))
            except Exception as exc:  # lỗi mạng / API: ghi lại và thử lại
                attempts.append({"raw": None, "error": f"client error: {type(exc).__name__}: {exc}"})
                continue
            try:
                action, rationale = parse_decision(raw, available_actions)
            except ValueError as exc:
                attempts.append({"raw": raw, "error": str(exc)})
                messages.append({"role": "assistant", "content": raw if isinstance(raw, str) else str(raw)})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your reply was invalid: {exc}. Reply again with only a JSON object "
                        f'{{"action_index": <0..{len(available_actions) - 1}>, "rationale": "..."}}.'
                    ),
                })
                continue
            attempts.append({"raw": raw, "error": None})
            return to_decision(action, rationale)

        if self.fallback_policy is not None:
            self.last_trace["fallback_used"] = True
            decision = self.fallback_policy.select_action(learner_state, available_actions, context)
            return PolicyDecision(
                action_type=decision.action_type,
                concept=decision.concept,
                difficulty=decision.difficulty,
                rationale=f"[fallback:{self.fallback_policy.name}] {decision.rationale}",
            )
        raise LLMDecisionError(
            f"LLM gave no valid action after {len(attempts)} attempt(s): {attempts[-1]['error'] if attempts else ''}"
        )
