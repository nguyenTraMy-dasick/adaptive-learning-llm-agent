"""Thành viên 3: sinh tập available_actions hợp lệ cho một bước ra quyết định.

Action space cố định về *loại* action (ActionType) và giống nhau cho cả ba policy;
các tham số cụ thể (concept, difficulty) phụ thuộc trạng thái phiên học:

- ASK(focus, d): với mỗi độ khó d còn câu chưa làm của focus concept.
- HINT(focus): khi đã có ít nhất một câu của focus concept được hỏi.
- EXPLAIN(focus): khi đã có focus concept.
- REVIEW_PREREQUISITE(p): với mỗi tiên quyết trực tiếp p của focus concept.
- ADVANCE(c): chuyển focus sang concept c khác nếu
    * c chưa phải focus và không phải tổ tiên của focus (quay lại tổ tiên là REVIEW),
    * mọi tiên quyết trực tiếp của c đã được giới thiệu,
    * c còn câu hỏi chưa làm.
  Khi phiên mới bắt đầu (chưa có focus) chỉ có ADVANCE tới các concept gốc.

Action space KHÔNG chặn theo ngưỡng mastery: policy có thể ADVANCE quá sớm, và đó
chính là một loại lỗi sư phạm cần đo ở Phase 5.

Thứ tự action trả về là cố định: ASK (Easy → Hard), HINT, EXPLAIN, REVIEW, ADVANCE;
REVIEW/ADVANCE xếp theo thứ tự topo của graph.
"""
from __future__ import annotations

from typing import Iterable

from adaptive_learning.policies.base import DecisionContext
from adaptive_learning.question_bank import QuestionBank
from adaptive_learning.schemas import Action, ActionType, Difficulty, LearnerState

DIFFICULTY_ORDER: tuple[str, ...] = (Difficulty.EASY.value, Difficulty.MEDIUM.value, Difficulty.HARD.value)


def introduced_from_attempts(question_bank: QuestionBank, attempted_ids: Iterable[str]) -> tuple[str, ...]:
    """Suy ra các concept đã học từ danh sách câu đã làm (giữ thứ tự xuất hiện)."""
    seen: dict[str, None] = {}
    for qid in attempted_ids:
        q = question_bank.get_question(qid)
        if q is not None:
            seen.setdefault(q.concept, None)
    return tuple(seen)


def build_available_actions(
    learner_state: LearnerState,
    question_bank: QuestionBank,
    context: DecisionContext | None = None,
) -> tuple[Action, ...]:
    """Tạo available_actions cho bước hiện tại.

    Args:
        learner_state: snapshot learner (dùng attempted_question_ids).
        question_bank: QuestionBank đã có concept_graph.
        context: focus concept và các concept đã giới thiệu. Nếu introduced_concepts
            rỗng sẽ suy ra từ attempted_question_ids.
    """
    graph = question_bank.concept_graph
    if graph is None:
        raise ValueError("QuestionBank must have a concept_graph to build the action space.")
    context = context or DecisionContext()
    attempted = set(learner_state.attempted_question_ids)
    topo = graph.topological_sort()

    focus = graph.canonicalize(context.focus_concept) if context.focus_concept else None
    if focus is not None and not graph.has_concept(focus):
        raise ValueError(f"Unknown focus concept: {context.focus_concept!r}")

    introduced = {graph.canonicalize(c) for c in context.introduced_concepts}
    if not introduced:
        introduced = set(introduced_from_attempts(question_bank, learner_state.attempted_question_ids))
    if focus is not None:
        introduced.add(focus)

    actions: list[Action] = []

    if focus is not None:
        for diff in DIFFICULTY_ORDER:
            if question_bank.get_questions(concept=focus, difficulty=diff, exclude_ids=attempted):
                actions.append(Action(ActionType.ASK, concept=focus, difficulty=diff))

        focus_questions = {q.question_id for q in question_bank.get_questions(concept=focus)}
        if focus_questions & attempted:
            actions.append(Action(ActionType.HINT, concept=focus))
        actions.append(Action(ActionType.EXPLAIN, concept=focus))

        prereqs = set(graph.get_prerequisites(focus))
        for concept in topo:
            if concept in prereqs:
                actions.append(Action(ActionType.REVIEW_PREREQUISITE, concept=concept))

    focus_ancestors = graph.get_ancestors(focus) if focus is not None else set()
    for concept in topo:
        if concept == focus or concept in focus_ancestors:
            continue
        if not set(graph.get_prerequisites(concept)) <= introduced:
            continue
        if not question_bank.get_questions(concept=concept, exclude_ids=attempted):
            continue
        actions.append(Action(ActionType.ADVANCE, concept=concept))

    return tuple(actions)
