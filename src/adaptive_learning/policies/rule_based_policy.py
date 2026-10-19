"""Thành viên 3: baseline luật (Baseline B — Rule-based Adaptive).

Luật dựa trên project_0 (Phase 4) và bổ sung xử lý prerequisite. Ký hiệu:
f = focus concept, m = mastery[f] (concept chưa có trong mastery dùng
`unknown_mastery`, mặc định 0.0), "vừa sai" = câu hỏi gần nhất thuộc f và trả lời sai.
Ngưỡng mặc định: low = 0.40, high = 0.75, advance = 0.85, weak_prerequisite = 0.60.

Luật áp dụng theo thứ tự, luật đầu tiên có action hợp lệ được chọn:

R0  Chưa có focus (đầu phiên)                    → ADVANCE tới concept đầu tiên (thứ tự topo).
R1  m ≥ advance                                   → ADVANCE (ưu tiên concept phụ thuộc trực tiếp
                                                    vào f và chưa thành thạo).
R2  vừa sai, m < low, có tiên quyết với
    mastery < weak_prerequisite                   → REVIEW_PREREQUISITE tiên quyết yếu nhất.
R3  vừa sai, m < low, bước trước không EXPLAIN f  → EXPLAIN(f).
R4  vừa sai, low ≤ m < high, bước trước không HINT → HINT(f).
R5  còn câu của f                                 → ASK(f, d) với d theo mastery:
                                                    m < low → Easy; low ≤ m < high → Medium;
                                                    m ≥ high → Hard. Nếu hết câu ở d thì lấy
                                                    độ khó gần nhất (hòa thì lấy dễ hơn).
R6  hết câu của f                                 → ADVANCE (như R1) → REVIEW tiên quyết yếu nhất
                                                    → EXPLAIN(f) → action đầu tiên.

Policy chỉ chọn trong available_actions và ghi luật đã dùng vào rationale (ví dụ "R3: ...").
"""
from __future__ import annotations

from dataclasses import dataclass

from adaptive_learning.policies.base import BasePolicy, DecisionContext, to_decision
from adaptive_learning.schemas import Action, ActionType, LearnerState, PolicyDecision

_DIFFICULTY_RANK = {"Easy": 0, "Medium": 1, "Hard": 2}


@dataclass(frozen=True)
class RuleConfig:
    low: float = 0.40
    high: float = 0.75
    advance: float = 0.85
    weak_prerequisite: float = 0.60
    unknown_mastery: float = 0.0

    def __post_init__(self) -> None:
        values = (self.low, self.high, self.advance, self.weak_prerequisite, self.unknown_mastery)
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError("All RuleConfig thresholds must be in [0, 1].")
        if not self.low <= self.high <= self.advance:
            raise ValueError("RuleConfig requires low <= high <= advance.")


class RuleBasedPolicy(BasePolicy):
    name = "rule_based"

    def __init__(self, config: RuleConfig | None = None, concept_graph=None) -> None:
        """
        Args:
            config: ngưỡng luật.
            concept_graph: ConceptGraph (tùy chọn) để R1 ưu tiên concept phụ thuộc
                trực tiếp vào focus. Không có graph thì giữ thứ tự của action space.
        """
        self.config = config or RuleConfig()
        self.concept_graph = concept_graph

    # ------------------------------------------------------------------ helpers
    def _mastery(self, state: LearnerState, concept: str | None) -> float:
        if concept is None:
            return self.config.unknown_mastery
        return float(state.mastery.get(concept, self.config.unknown_mastery))

    @staticmethod
    def _of_type(actions: tuple[Action, ...], action_type: ActionType) -> list[Action]:
        return [a for a in actions if a.action_type == action_type]

    def _target_difficulty(self, m: float) -> str:
        if m < self.config.low:
            return "Easy"
        if m < self.config.high:
            return "Medium"
        return "Hard"

    def _pick_advance(self, state: LearnerState, actions: tuple[Action, ...], focus: str | None) -> Action | None:
        candidates = self._of_type(actions, ActionType.ADVANCE)
        if not candidates:
            return None
        dependents = set(self.concept_graph.get_dependents(focus)) if (self.concept_graph and focus) else set()

        def key(item: tuple[int, Action]) -> tuple[int, int, int]:
            index, action = item
            not_mastered = self._mastery(state, action.concept) < self.config.advance
            return (0 if not_mastered else 1, 0 if action.concept in dependents else 1, index)

        return min(enumerate(candidates), key=key)[1]

    def _weakest_prerequisite(
        self, state: LearnerState, actions: tuple[Action, ...], only_weak: bool
    ) -> Action | None:
        reviews = self._of_type(actions, ActionType.REVIEW_PREREQUISITE)
        if only_weak:
            reviews = [a for a in reviews if self._mastery(state, a.concept) < self.config.weak_prerequisite]
        if not reviews:
            return None
        return min(enumerate(reviews), key=lambda item: (self._mastery(state, item[1].concept), item[0]))[1]

    def _pick_ask(self, actions: tuple[Action, ...], focus: str, target: str) -> Action | None:
        asks = [a for a in self._of_type(actions, ActionType.ASK) if a.concept == focus]
        if not asks:
            return None
        goal = _DIFFICULTY_RANK[target]
        return min(asks, key=lambda a: (abs(_DIFFICULTY_RANK[a.difficulty] - goal), _DIFFICULTY_RANK[a.difficulty]))

    @staticmethod
    def _find(actions: tuple[Action, ...], action_type: ActionType, concept: str | None) -> Action | None:
        for a in actions:
            if a.action_type == action_type and a.concept == concept:
                return a
        return None

    @staticmethod
    def _last_was(context: DecisionContext, action_type: ActionType, concept: str | None) -> bool:
        last = context.last_action
        return last is not None and last.action_type == action_type and last.concept == concept

    # --------------------------------------------------------------- decision
    def select_action(
        self,
        learner_state: LearnerState,
        available_actions: tuple[Action, ...],
        context: DecisionContext | None = None,
    ) -> PolicyDecision:
        self._require_actions(available_actions)
        context = context or DecisionContext()
        cfg = self.config
        focus = context.focus_concept

        if focus is None:
            advance = self._pick_advance(learner_state, available_actions, None)
            if advance:
                return to_decision(advance, "R0: start of session, introduce the first concept")
            return to_decision(available_actions[0], "R0: no focus and no ADVANCE available, first action")

        m = self._mastery(learner_state, focus)
        wrong = context.last_answer_wrong_on_focus

        if m >= cfg.advance:
            advance = self._pick_advance(learner_state, available_actions, focus)
            if advance:
                return to_decision(advance, f"R1: mastery {m:.2f} >= {cfg.advance}, advance")

        if wrong and m < cfg.low:
            review = self._weakest_prerequisite(learner_state, available_actions, only_weak=True)
            if review:
                pm = self._mastery(learner_state, review.concept)
                return to_decision(
                    review, f"R2: wrong answer, mastery {m:.2f} < {cfg.low}, weak prerequisite ({pm:.2f})"
                )
            explain = self._find(available_actions, ActionType.EXPLAIN, focus)
            if explain and not self._last_was(context, ActionType.EXPLAIN, focus):
                return to_decision(explain, f"R3: wrong answer, mastery {m:.2f} < {cfg.low}, explain concept")

        if wrong and cfg.low <= m < cfg.high:
            hint = self._find(available_actions, ActionType.HINT, focus)
            if hint and not self._last_was(context, ActionType.HINT, focus):
                return to_decision(hint, f"R4: wrong answer, mastery {m:.2f} in [{cfg.low}, {cfg.high}), give hint")

        target = self._target_difficulty(m)
        ask = self._pick_ask(available_actions, focus, target)
        if ask:
            note = "" if ask.difficulty == target else f" (no {target} left)"
            return to_decision(ask, f"R5: mastery {m:.2f} -> ask {ask.difficulty}{note}")

        advance = self._pick_advance(learner_state, available_actions, focus)
        if advance:
            return to_decision(advance, "R6: no questions left for focus, advance")
        review = self._weakest_prerequisite(learner_state, available_actions, only_weak=False)
        if review:
            return to_decision(review, "R6: no questions left for focus, review prerequisite")
        explain = self._find(available_actions, ActionType.EXPLAIN, focus)
        if explain:
            return to_decision(explain, "R6: no questions left for focus, explain")
        return to_decision(available_actions[0], "R6: fallback to first available action")
