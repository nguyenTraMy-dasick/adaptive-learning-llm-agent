"""Thành viên 3: learner mô phỏng cho Phase 6 (so sánh hiệu quả học giữa các policy).

Learner có "kiến thức thật" k[c] ∈ [0, 1] cho từng concept. Policy KHÔNG thấy k; policy chỉ
thấy mastery do StudentModel (Thành viên 2) ước lượng từ các câu trả lời.

Giả định của mô hình (mọi tham số nằm trong data/simulated_learners.json):

1. Sẵn sàng (readiness): r(c) = min k[p] trên các tiên quyết trực tiếp p của c (1.0 nếu không có).
   Hệ số tiên quyết: g(c) = (1 - w) + w * r(c), với w = prerequisite_weight.
2. Trả lời câu hỏi: hiệu quả e = clamp(k[c] * g(c) * factor[difficulty]),
   factor Easy = 1.2, Medium = 1.0, Hard = 0.8 (kiến thức 0 thì chỉ còn đoán, ở mọi độ khó).
   P(đúng) = guess + (1 - guess - slip) * e. Sai thì chọn ngẫu nhiên một phương án sai.
3. Học (cập nhật k sau mỗi lượt hợp lệ, theo action đã thực thi):
   ASK (làm câu + xem giải thích)  k[c] += practice_gain * (1 - k[c]) * g(c)
   EXPLAIN c                       k[c] += explain_gain  * (1 - k[c]) * g(c)
   HINT c                          k[c] += hint_gain     * (1 - k[c]) * g(c)
   REVIEW_PREREQUISITE p           k[p] += review_gain   * (1 - k[p]) * g(p)
   ADVANCE                         không đổi
4. Pre/post-test: điểm kỳ vọng = trung bình P(đúng) trên toàn bộ question bank, tính trực
   tiếp từ k (không bốc thăm, không "dùng hết" câu hỏi).

Kết quả Phase 6 vì vậy phản ánh các giả định trên, không thay thế thí nghiệm với người học thật.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from adaptive_learning.question_bank import ConceptGraph, QuestionBank
from adaptive_learning.schemas import ActionType, Question, TurnResult

DIFFICULTY_FACTOR = {"Easy": 1.2, "Medium": 1.0, "Hard": 0.8}


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


@dataclass(frozen=True)
class LearnerProfile:
    id: str
    description: str = ""
    initial_knowledge: Mapping[str, float] = field(default_factory=dict)
    default_knowledge: float = 0.05
    practice_gain: float = 0.10
    explain_gain: float = 0.15
    hint_gain: float = 0.07
    review_gain: float = 0.15
    prerequisite_weight: float = 0.5
    guess: float = 0.25
    slip: float = 0.10

    def __post_init__(self) -> None:
        values = [self.default_knowledge, self.practice_gain, self.explain_gain, self.hint_gain,
                  self.review_gain, self.prerequisite_weight, self.guess, self.slip,
                  *self.initial_knowledge.values()]
        if not all(0.0 <= float(v) <= 1.0 for v in values):
            raise ValueError(f"Profile {self.id!r}: every parameter must be in [0, 1].")
        if self.guess + self.slip >= 1.0:
            raise ValueError(f"Profile {self.id!r}: guess + slip must be < 1.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LearnerProfile:
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown learner profile fields: {sorted(unknown)}")
        return cls(**{**data, "initial_knowledge": dict(data.get("initial_knowledge", {}))})


def load_profiles(path: str | Path, graph: ConceptGraph | None = None) -> list[LearnerProfile]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data["learners"] if isinstance(data, dict) else data
    profiles = [LearnerProfile.from_dict(item) for item in items]
    if graph is not None:
        for p in profiles:
            for concept in p.initial_knowledge:
                if not graph.has_concept(concept):
                    raise ValueError(f"Profile {p.id!r}: unknown concept {concept!r}")
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError("Learner profile ids must be unique.")
    return profiles


class SimulatedLearner:
    """Một learner mô phỏng; dùng `respond` làm Responder và `observe` làm on_turn của engine."""

    def __init__(self, profile: LearnerProfile, question_bank: QuestionBank, seed: int | None = None) -> None:
        if question_bank.concept_graph is None:
            raise ValueError("QuestionBank must have a concept_graph.")
        self.profile = profile
        self.question_bank = question_bank
        self.graph = question_bank.concept_graph
        self.rng = random.Random(seed)
        self.knowledge: dict[str, float] = {
            c: float(profile.initial_knowledge.get(c, profile.default_knowledge))
            for c in self.graph.all_concepts()
        }

    # ------------------------------------------------------------ mô hình
    def readiness(self, concept: str) -> float:
        prereqs = self.graph.get_prerequisites(concept)
        return min((self.knowledge[p] for p in prereqs), default=1.0)

    def _gate(self, concept: str) -> float:
        w = self.profile.prerequisite_weight
        return (1.0 - w) + w * self.readiness(concept)

    def p_correct(self, question: Question) -> float:
        effective = _clamp(self.knowledge[question.concept] * self._gate(question.concept)
                           * DIFFICULTY_FACTOR.get(question.difficulty, 1.0))
        return self.profile.guess + (1.0 - self.profile.guess - self.profile.slip) * effective

    def _learn(self, concept: str | None, gain: float) -> None:
        if concept is None or concept not in self.knowledge or gain <= 0:
            return
        k = self.knowledge[concept]
        self.knowledge[concept] = _clamp(k + gain * (1.0 - k) * self._gate(concept))

    # --------------------------------------------------- giao tiếp với engine
    def respond(self, question: Question) -> str:
        if self.rng.random() < self.p_correct(question):
            return question.correct_answer
        wrong = [letter for letter in "ABCD" if letter != question.correct_answer]
        return self.rng.choice(wrong)

    def observe(self, result: TurnResult) -> None:
        """Cập nhật kiến thức theo action đã thực thi (bỏ qua lượt invalid)."""
        if not result.valid or result.decision is None:
            return
        action, concept = result.decision.action_type, result.decision.concept
        p = self.profile
        if action == ActionType.ASK and result.question is not None:
            self._learn(result.question.concept, p.practice_gain)
        elif action == ActionType.EXPLAIN:
            self._learn(concept, p.explain_gain)
        elif action == ActionType.HINT:
            self._learn(concept, p.hint_gain)
        elif action == ActionType.REVIEW_PREREQUISITE:
            self._learn(concept, p.review_gain)

    # --------------------------------------------------------- đo lường
    def expected_test_score(self) -> float:
        """Điểm kỳ vọng (0–1) trên toàn bộ question bank với kiến thức hiện tại."""
        questions = self.question_bank.get_all_questions()
        return sum(self.p_correct(q) for q in questions) / len(questions)

    def concept_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for concept in self.graph.topological_sort():
            qs = self.question_bank.get_questions(concept=concept)
            if qs:
                scores[concept] = sum(self.p_correct(q) for q in qs) / len(qs)
        return scores
