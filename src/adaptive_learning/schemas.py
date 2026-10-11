"""HỢP ĐỒNG CHUNG — xem TEAM_TASKS.md trước khi thay đổi schema.

Cả ba policy và engine phải dùng chung các kiểu dưới đây. Hãy giữ schema nhỏ,
serializable, không đưa logic chọn policy/chấm điểm vào file này.
TODO (thành viên phụ trách):
- Chốt cách validate mastery trong khoảng [0, 1].
- Chốt cách biểu diễn danh sách (tuple/list) khi parse JSON.
- Giữ difficulty thống nhất với CSV: Easy, Medium, Hard.
- Giữ prerequisite_concept đúng tên cột hiện tại, trừ khi có migration được duyệt.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ActionType(str, Enum):
    """Các action chung mà policy được phép đề xuất."""
    ASK = "ASK"
    HINT = "HINT"
    EXPLAIN = "EXPLAIN"
    REVIEW_PREREQUISITE = "REVIEW_PREREQUISITE"
    ADVANCE = "ADVANCE"


@dataclass(frozen=True)
class LearnerState:
    """Snapshot learner state; policy không được mutate object này."""
    mastery: Mapping[str, float] = field(default_factory=dict)
    recent_errors: tuple[str, ...] = ()
    recent_actions: tuple[str, ...] = ()
    attempted_question_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Action:
    """Action hợp lệ được engine tạo trước khi gọi policy."""
    action_type: ActionType
    concept: str | None = None
    difficulty: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class PolicyDecision:
    """Output thống nhất của cả ba policy; rationale chỉ phục vụ phân tích."""
    action_type: ActionType
    concept: str | None = None
    difficulty: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class TurnResult:
    """Kết quả khái quát một lượt; chốt fields cuối cùng khi tích hợp."""
    decision: PolicyDecision
    valid: bool
    question_id: str | None = None
    question: Any = None
    feedback: Any = None
    correct: bool | None = None
    updated_state: LearnerState | None = None
    error: str | None = None
