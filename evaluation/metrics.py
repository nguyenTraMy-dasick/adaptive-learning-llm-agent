"""Thành viên 3: các metric đánh giá policy.

Hai nhóm:
1. Chất lượng quyết định (Phase 5) — so với nhãn expert trong decision_scenarios.json:
   - valid_rate: tỉ lệ quyết định nằm trong available_actions.
   - exact_agreement: quyết định trùng một "preferred action" của annotator.
   - acceptable_agreement: trùng preferred hoặc acceptable.
   - type_agreement: cùng loại action (ASK/HINT/...) với một preferred action.
   - auto_flags: cờ heuristic (xem FlagConfig). Các cờ này dùng giả định gần với
     rule-based policy nên CHỈ để tham khảo; thước đo chính là nhãn expert.
2. Hiệu quả học (Phase 6): learning_gain, normalized_gain, efficiency.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


# ----------------------------------------------------------- decision quality
def _key(action: Mapping[str, Any] | None) -> tuple[Any, Any, Any] | None:
    if not action:
        return None
    return (action.get("action_type"), action.get("concept"), action.get("difficulty"))


def label_agreement(decision: Mapping[str, Any] | None, label: Mapping[str, Any]) -> dict[str, bool]:
    """So một quyết định (dict action) với một nhãn annotator."""
    key = _key(decision)
    preferred = {_key(a) for a in label.get("preferred_actions", [])}
    acceptable = preferred | {_key(a) for a in label.get("acceptable_actions", [])}
    preferred_types = {k[0] for k in preferred if k}
    return {
        "exact": key in preferred,
        "acceptable": key in acceptable,
        "type": key is not None and key[0] in preferred_types,
    }


def scenario_agreement(decision: Mapping[str, Any] | None, labels: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    """Trung bình độ đồng thuận qua các annotator có preferred_actions; None nếu chưa có nhãn."""
    usable = [lab for lab in labels if lab.get("preferred_actions")]
    if not usable:
        return None
    scores = [label_agreement(decision, lab) for lab in usable]
    return {k: sum(s[k] for s in scores) / len(scores) for k in ("exact", "acceptable", "type")}


@dataclass(frozen=True)
class FlagConfig:
    """Ngưỡng cho cờ heuristic. Cố ý khác ngưỡng của RuleConfig."""

    premature_advance_below: float = 0.6
    too_hard_below: float = 0.4
    too_easy_at_or_above: float = 0.85
    unknown_mastery: float = 0.0


def auto_flags(
    decision: Mapping[str, Any] | None,
    mastery: Mapping[str, float],
    context: Mapping[str, Any],
    available_actions: Sequence[Mapping[str, Any]],
    config: FlagConfig = FlagConfig(),
) -> list[str]:
    """Các lỗi sư phạm phát hiện được tự động (danh sách tên cờ)."""
    if not decision:
        return ["no_decision"]
    flags: list[str] = []
    focus = context.get("focus_concept")
    m = float(mastery.get(focus, config.unknown_mastery)) if focus else None
    action_type = decision.get("action_type")
    focus_has_questions = any(a.get("action_type") == "ASK" and a.get("concept") == focus for a in available_actions)

    if action_type == "ADVANCE" and m is not None and m < config.premature_advance_below and focus_has_questions:
        flags.append("premature_advance")
    if action_type == "ASK" and m is not None:
        if decision.get("difficulty") == "Hard" and m < config.too_hard_below:
            flags.append("too_hard")
        if decision.get("difficulty") == "Easy" and m >= config.too_easy_at_or_above:
            flags.append("too_easy")
    last = context.get("last_action") or {}
    if action_type in {"HINT", "EXPLAIN"} and _key(last) == _key(decision):
        flags.append("repeated_support")
    return flags


def summarize_decisions(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Tổng hợp cho một policy.

    Mỗi record cần: valid (bool), decision (dict|None), agreement (dict|None), flags (list).
    """
    records = list(records)
    n = len(records)
    if n == 0:
        return {"n": 0}
    types = Counter((r.get("decision") or {}).get("action_type", "NONE") for r in records)
    labeled = [r["agreement"] for r in records if r.get("agreement")]
    flags = Counter(f for r in records for f in r.get("flags", []))
    summary: dict[str, Any] = {
        "n": n,
        "valid_rate": sum(bool(r.get("valid")) for r in records) / n,
        "action_type_distribution": {k: v / n for k, v in sorted(types.items())},
        "flag_rate": {k: v / n for k, v in sorted(flags.items())},
        "labeled_scenarios": len(labeled),
    }
    if labeled:
        for k in ("exact", "acceptable", "type"):
            summary[f"{k}_agreement"] = sum(a[k] for a in labeled) / len(labeled)
    return summary


# ----------------------------------------------------------- learning outcome
def learning_gain(pre: float, post: float) -> float:
    """LearningGain = PostTest - PreTest (project_0, Phase 6)."""
    return post - pre


def normalized_gain(pre: float, post: float, max_score: float = 1.0) -> float | None:
    """Hake's normalized gain (post - pre) / (max - pre); None nếu pre đã tối đa."""
    if max_score - pre <= 0:
        return None
    return (post - pre) / (max_score - pre)


def efficiency(gain: float, n_questions: int) -> float | None:
    """Efficiency = LearningGain / #Questions; None nếu chưa hỏi câu nào."""
    if n_questions <= 0:
        return None
    return gain / n_questions
